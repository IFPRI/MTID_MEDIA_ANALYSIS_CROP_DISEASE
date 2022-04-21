from gatenlp.gateworker import GateWorker
from gatenlp.gateworker import GateWorkerAnnotator
from gatenlp import Document
from gatenlp.pam.pampac import Ann, AnnAt, Rule, Pampac, AddAnn, N, Seq, Or
from gatenlp.pam.matcher import FeatureMatcher, IfNot
import glob
import argparse
import os
import json
from XSModelManager.ModelManager import ModelManager
from XSNLPReader.readerPostProcessor import BertPostProcessor as DataPostProcessor
from configobj import ConfigObj
import csv
from downloadWebPage import WebDownload

class DummyReader:
    def __init__(self, sample, readerPostProcessor):
        self.sample = sample
        self.readerPostProcessor = readerPostProcessor
        #print(self.readerPostProcessor)

    def _reset_iter(self):
        pass
    def __iter__(self):
        return self
    def __next__(self):
        return self.readerPostProcessor.postProcess4Model(self.sample)
    def __len__(self):
        return 1


def get_ann_counts(anns, doc_text):
    count_dict = {}
    for each_ann in anns:
        text = doc_text[each_ann.start:each_ann.end].lower()
        if text not in count_dict:
            count_dict[text] = 0
        count_dict[text] += 1
    return count_dict


class IFPRI_Disease:
    def __init__(self, model):
        self.readerPostProcessor = DataPostProcessor(['text'], 'label', config=config)
        self.mm = ModelManager(gpu=args.gpu, config=config)
        self.mm.load_model(model)

        self.gs = GateWorker()
        self.gs_app = GateWorkerAnnotator('gateapp/IFPRI_Diease.xgapp', self.gs)

        self.webManager = WebDownload()

    def apply_single_doc(self, doc_in):
        gdoc = self.gs.loadDocumentFromFile(doc_in)
        pdoc = self.gs.gdoc2pdoc(gdoc)
        doc_text = pdoc.text
        all_results = self.apply_to_single_input(doc_text, pdoc)
        self.gs.deleteResource(gdoc)

    def apply_single_url(self, url):
        title, html, raw_html, clean_text = self.webManager.downLoadWebPage(url)
        print(title)
        #print(clean_text)
        title = str(title)
        doc_text = str(title)+'\n'+clean_text
        pdoc = Document(doc_text)
        all_results = self.apply_to_single_input(doc_text, pdoc)
        output_json_for_url = {
                'url':url,
                'date_of_the_artical':None,
                'title':title,
                'items':all_results
                }
        return output_json_for_url


    def get_names(self, annotation):
        name_dict = {}
        if annotation:
            anno_feature = annotation.features
            anno_value = anno_feature.get('value')
            common_name = anno_feature.get('Common_Name')
            scientific_name = anno_feature.get('Scientific_Name')
            if not common_name and not scientific_name:
                name_dict['Common_name'] = None
                name_dict['Scientific_name'] = anno_value
            elif common_name:
                name_dict['Common_name'] = common_name
                name_dict['Scientific_name'] = anno_value
            elif scientific_name:
                name_dict['Common_name'] = anno_value
                name_dict['Scientific_name'] = scientific_name
        else:
            name_dict['Common_name'] = None
            name_dict['Scientific_name'] = None
        return name_dict



    def apply_to_single_input(self, doc_text, pdoc):
        process_doc = self.gs_app(pdoc)
        all_anns = process_doc.annset()
        sentence_anns = all_anns.with_type("Sentence")
        eppo_animal_anns = all_anns.with_type("EPPO_animals")
        plant_anns = all_anns.with_type("Plant")
        disease_anns = all_anns.with_type("PlantDisease")
        
        all_results = []

        for each_ann in sentence_anns:
            sample = {}
            text = doc_text[each_ann.start:each_ann.end]
            sample['text'] = text
            sample['label'] = None
            dmreader = DummyReader(sample, self.readerPostProcessor)
            #print(dmreader.readerPostProcessor)
            oo = self.mm.apply(dmreader, batch_size=1)
            #print(oo['all_pred_label_string'])
            dmg_type = oo['all_pred_label_string'][0]
            if dmg_type != 'neg':
                cl_pest_ann,cl_host,cl_disease = self.get_related_info(each_ann.start, each_ann.end, eppo_animal_anns, plant_anns, disease_anns)
                disease_name_dict = self.get_names(cl_disease)
                disease_name_dict['local_name'] = None
                pest_name_dict = self.get_names(cl_pest_ann)
                host_name_dict = self.get_names(cl_host)

                output_json = {
                        'Disease':disease_name_dict,
                        'Host':host_name_dict,
                        'Pest':pest_name_dict,
                        'Impacted_area':{
                            'Country':None,
                            'Sub-region':None,
                            'City':None,
                            },
                        'Type_of_impact':dmg_type,
                        'Duration':{
                            'start':None,
                            'end':None,
                            },
                        'Origin_country':None,
                        'Orign_Sentence':text
                        }
                print(output_json)
                all_results.append(output_json)
        return all_results


    def find_closest_anno(self, sent_start, send_end, current_annotation):
        distance = 99999
        look_ahead_thres = 1000
        output_anno = None
        for each_ann in current_annotation:
            each_ann_start = each_ann.start
            each_ann_end = each_ann.end
            each_ann_mid = (each_ann_end - each_ann_start)/2 + each_ann_start
            #current_dis = min(abs(sent_start - each_ann_mid), abs(send_end - each_ann_mid))
            #current_dis = abs((sent_start - each_ann_mid) + (send_end - each_ann_mid))
            if each_ann_mid > sent_start and each_ann_mid < send_end:
                distance = 0
                output_anno = each_ann
                break
            else:
                abs_dis = (each_ann_mid - sent_start) + (each_ann_mid - send_end)
                if abs_dis < 0 and abs(abs_dis) < 1000:
                    current_dis = abs(abs_dis)
                    if current_dis < distance:
                        distance = current_dis
                        output_anno = each_ann
                else:
                    if output_anno:
                        break
                    else:
                        current_dis = abs(abs_dis)
                        if current_dis < distance:
                            distance = current_dis
                            output_anno = each_ann

        #print(sent_start, send_end, output_anno)
        return output_anno
            

    def get_related_info(self, sent_start, send_end, eppo_animal_anns, plant_anns, disease_anns):
        cl_pest_ann = self.find_closest_anno(sent_start, send_end, eppo_animal_anns)
        cl_host = self.find_closest_anno(sent_start, send_end, plant_anns)
        cl_disease = self.find_closest_anno(sent_start, send_end, disease_anns)
        return cl_pest_ann, cl_host, cl_disease



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputFolder", help="input folder")
    parser.add_argument("--inputURL", help="input url")
    parser.add_argument("--model", help="path to trained model")
    parser.add_argument("--gpu", help="use gpu", default=False, action='store_true')
    parser.add_argument("--config", help="config file")
    parser.add_argument("--outputJson", help="json output", default='diseaseOutput.json')
    parser.add_argument("--outputTsv", help="disease type tsv output", default='diseaseOutput.tsv')
    parser.add_argument("--outputSummary", help="summary output", default='diseaseSummaryOutput.tsv')
    args = parser.parse_args()

    config = ConfigObj(args.config)

    disease_manager = IFPRI_Disease(args.model)
    output_json = {}
    if args.inputFolder:
        all_input_list = glob.glob(os.path.join(args.inputFolder, '*'))
        for each_file in all_input_list:
            print(each_file)
            disease_manager.apply_single_doc(each_file)
            break
    if args.inputURL:
        output_json = [disease_manager.apply_single_url(args.inputURL)]

    if args.outputJson:
        print(output_json)
        with open(args.outputJson, 'w') as fo:
            json.dump(output_json, fo)








    ##readerPostProcessor = DataPostProcessor(['text'], 'label', config=config)
    ##mm = ModelManager(gpu=args.gpu, config=config)
    ##mm.load_model(args.model)

    ##gs = GateWorker()
    ##gs_app = GateWorkerAnnotator('gateapp/IFPRI_Diease.xgapp', gs)
    #all_input_list = glob.glob(os.path.join(args.inputFolder, '*'))

    #fo = open(args.outputTsv, 'w')
    #tsv_line = 'file\tstart\tend\tlabel\tsentence\n'
    #fo.write(tsv_line)

    #doc_counts = {}
    #for each_file in all_input_list:
    #    print(each_file)
    #    doc_counts[each_file] = {}
    #    doc_counts[each_file]['damage_type'] = {}

    #    gdoc = gs.loadDocumentFromFile(each_file)
    #    pdoc = gs.gdoc2pdoc(gdoc)
    #    doc_text = pdoc.text
    #    process_doc = gs_app(pdoc)
    #    all_anns = process_doc.annset()
    #    sentence_anns = all_anns.with_type("Sentence")
    #    eppo_animal_anns = all_anns.with_type("EPPO_animals")
    #    plant_anns = all_anns.with_type("Plant")
    #    disease = all_anns.with_type("PlantDisease")

    #    doc_counts[each_file]['plants'] = get_ann_counts(plant_anns, doc_text)
    #    doc_counts[each_file]['disease'] = get_ann_counts(disease, doc_text)
    #    doc_counts[each_file]['pest'] = get_ann_counts(eppo_animal_anns, doc_text)


    #    for each_ann in sentence_anns:
    #        sample = {}
    #        text = doc_text[each_ann.start:each_ann.end]
    #        sample['text'] = text
    #        sample['label'] = None
    #        dmreader = DummyReader(sample, readerPostProcessor)
    #        oo = mm.apply(dmreader, batch_size=1)
    #        #print(oo['all_pred_label_string'])
    #        dmg_type = oo['all_pred_label_string'][0]
    #        if dmg_type != 'neg':
    #            #tsv_list = [each_file, str(each_ann.start), str(each_ann.end), dmg_type, text]
    #            #print(tsv_list)
    #            #tsv_line = '\t'.join(tsv_list)
    #            #fo.write(tsv_line+'\n')
    #            #if dmg_type not in doc_counts[each_file]['damage_type']:
    #            #    doc_counts[each_file]['damage_type'][dmg_type] = 0
    #            #doc_counts[each_file]['damage_type'][dmg_type] += 1

    #    gs.deleteResource(gdoc)
    #fo.close()

    #csv_keys = []
    #for each_file in doc_counts:
    #    for each_output_type in doc_counts[each_file]:
    #        for each_item in doc_counts[each_file][each_output_type]:
    #            csv_keys.append(' ||| '.join([each_output_type, each_item]))

    #f_csv = open(args.outputSummary, 'w', newline='')
    #spamwriter = csv.writer(f_csv, delimiter='\t',quotechar='"', quoting=csv.QUOTE_MINIMAL)
    #csv_keys = list(set(csv_keys))
    #print(csv_keys)
    #csv_headings = ['file_name']+csv_keys
    #spamwriter.writerow(csv_headings)

    #for each_file in doc_counts:
    #    csv_line = [each_file]
    #    for each_key in csv_keys:
    #        each_key_tok = each_key.split(' ||| ')
    #        output_type = each_key_tok[0]
    #        output_item = each_key_tok[1]
    #        have_value = False
    #        if output_type in doc_counts[each_file]:
    #            if output_item in doc_counts[each_file][output_type]:
    #                csv_line.append(doc_counts[each_file][output_type][output_item])
    #                have_value = True
    #        if not have_value:
    #            csv_line.append(0)
    #    print(csv_line)
    #    spamwriter.writerow(csv_line)
    #f_csv.close()







