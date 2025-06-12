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
import re
import feedparser
import openai
import signal
from pathlib import Path
script_path = os.path.abspath(__file__)
print(script_path)
parent = os.path.dirname(script_path)

def timeout_handler(signum, frame):
    raise TimeoutError()

signal.signal(signal.SIGALRM, timeout_handler)


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



class IFPRI_Disease_Base:
    def __init__(self):
        self.webManager = WebDownload()

    def apply_single_url(self, url, save_html=None, date=None, save_txt=None):
        title = None
        try:
            signal.alarm(120)
            title, html, raw_html, clean_text = self.webManager.downLoadWebPage(url)
            signal.alarm(0)
            #print(title)
            #print(clean_text)
            title = str(title)
            doc_text = str(title)+'\n'+clean_text
            pdoc = Document(doc_text)
            all_results = self.apply_to_single_input(doc_text, pdoc)
            if save_html:
                with open(save_html, 'w') as fo:
                    fo.write(html)

            if save_txt:
                with open(save_txt, 'w') as fo:
                    fo.write(doc_text)
        except Exception as e:
            print("The function timed out!")
            signal.alarm(0)
            all_results = []


        output_json_for_url = {
                'url':url,
                'date_of_the_artical':date,
                'title':title,
                'items':all_results,
                'txt_file':save_txt
                }

        return output_json_for_url

class IFPRI_DiseaseGPT(IFPRI_Disease_Base):
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        openai.api_key = config['OPENAI']['key']
        self.assitant_prompt = config['OPENAI']['prompt']

    def getGPTresponse(self, content):
        messages = [
            {"role": "assistant", "content":self.assitant_prompt},
            {"role": "user", "content":content}
        ]
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=messages
        )
        return response

    def apply_to_single_input(self, doc_text, pdoc):
        all_results = []
        max_retry = 2
        num_retry = 0
        timeout = 10
        finish = False

        #set default values:
        output_json = {
                   "start_offset":None,
                   "end_offset":None,
                   "Disease":None,
                   "Host":None,
                   "Pest":None,
                   "Impacted_area":{
                       'Country':None,
                       'Sub-region':None,
                       'City':None,
                       },
                   "Type_of_impact":None,
                   "Duration":{
                       "start":None,
                       "end":None,
                       },
                   "Origin_country":None,
                   "Orign_Sentence":None
                   }


        while (num_retry < max_retry) and (not finish):
            try: 
            #if True:
                signal.alarm(timeout)
                num_retry += 1
                response = self.getGPTresponse(doc_text)
                signal.alarm(0)
                output_json = json.loads(response.choices[0].message.content)
                print(output_json)
                finish = True
                all_results = [output_json]
            except Exception as e:
                signal.alarm(0)
                print('error:',e)
                if num_retry == max_retry:
                    print('max retry reached')
        print('finished: ', all_results)

        return all_results


class IFPRI_Disease(IFPRI_Disease_Base):
    def __init__(self, model, **kwargs):
        super().__init__(**kwargs)
        self.readerPostProcessor = DataPostProcessor(['text'], 'label', config=config)
        self.mm = ModelManager(gpu=args.gpu, config=config)
        self.mm.load_model(model)

        self.gs = GateWorker()
        self.gs_app = GateWorkerAnnotator(os.path.join(parent,'gateapp/IFPRI_Diease.xgapp'), self.gs)

        #self.webManager = WebDownload()

    def apply_single_doc(self, doc_in):
        gdoc = self.gs.loadDocumentFromFile(doc_in)
        pdoc = self.gs.gdoc2pdoc(gdoc)
        doc_text = pdoc.text
        all_results = self.apply_to_single_input(doc_text, pdoc)
        self.gs.deleteResource(gdoc)


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
        lookup_anns = all_anns.with_type("Lookup")
        date_anns = all_anns.with_type("Date")
        
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
                cl_pest_ann,cl_host,cl_disease, cl_country, cl_date = self.get_related_info(each_ann.start, each_ann.end, eppo_animal_anns, plant_anns, disease_anns, lookup_anns, date_anns)
                if cl_date:
                    cl_date_str = doc_text[cl_date.start:cl_date.end]
                else:
                    cl_date_str = None
                disease_name_dict = self.get_names(cl_disease)
                disease_name_dict['local_name'] = None
                pest_name_dict = self.get_names(cl_pest_ann)
                host_name_dict = self.get_names(cl_host)
                country_iso3 = None
                if cl_country:
                    country_iso3 = cl_country.features.get('ISO3')

                output_json = {
                        'start_offset':each_ann.start,
                        'end_offset':each_ann.end,
                        'Disease':disease_name_dict,
                        'Host':host_name_dict,
                        'Pest':pest_name_dict,
                        'Impacted_area':{
                            'Country':country_iso3,
                            'Sub-region':None,
                            'City':None,
                            },
                        'Type_of_impact':dmg_type,
                        'Duration':{
                            'start':cl_date_str,
                            'end':None,
                            },
                        'Origin_country':country_iso3,
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


    def find_closest_anno_with_feature(self, sent_start, send_end, current_annotation, feature, value):
        distance = 99999
        look_ahead_thres = 1000
        output_anno = None
        for each_ann in current_annotation:
            #print(each_ann.features)
            if each_ann.features.get(feature) == value:
                #print(each_ann.features)
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

            

    def get_related_info(self, sent_start, send_end, eppo_animal_anns, plant_anns, disease_anns, lookup_anns, date_anns):
        cl_pest_ann = self.find_closest_anno(sent_start, send_end, eppo_animal_anns)
        cl_host = self.find_closest_anno(sent_start, send_end, plant_anns)
        cl_disease = self.find_closest_anno(sent_start, send_end, disease_anns)
        cl_country = self.find_closest_anno_with_feature(sent_start, send_end, lookup_anns, 'minorType', 'country')
        cl_date = self.find_closest_anno(sent_start, send_end, date_anns)
        
        return cl_pest_ann, cl_host, cl_disease, cl_country, cl_date



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputFolder", help="input folder")
    parser.add_argument("--inputURL", help="input url")
    parser.add_argument("--model", help="path to trained model")
    parser.add_argument("--gpu", help="use gpu", default=False, action='store_true')
    parser.add_argument("--rss", help="use rss from config file", default=False, action='store_true')
    parser.add_argument("--config", help="config file")
    parser.add_argument("--useGPT", help="use GPT", default=False, action='store_true')
    parser.add_argument("--outputFolder", help="outputFolder")
    parser.add_argument("--outputJson", help="json output", default='diseaseOutput.json')
    parser.add_argument("--outputTsv", help="disease type tsv output", default='diseaseOutput.tsv')
    parser.add_argument("--outputSummary", help="summary output", default='diseaseSummaryOutput.tsv')
    args = parser.parse_args()

    config = ConfigObj(args.config)

    if args.useGPT:
        disease_manager = IFPRI_DiseaseGPT(config)
        backup_manager = IFPRI_Disease(args.model)
    else:
        disease_manager = IFPRI_Disease(args.model)
        backup_manager = disease_manager
    output_json = {}

    if args.inputURL:
        output_json = [disease_manager.apply_single_url(args.inputURL)]#
        if args.outputJson:
            print(output_json)
            with open(args.outputJson, 'w') as fo:
                json.dump(output_json, fo)


    elif args.rss:
        rss_url = config['RSS_FEED']['url']
        NewsFeed = feedparser.parse(rss_url)
        date = NewsFeed['headers']['date']
        date = date.replace(',','')
        date = date.replace(' ','_')
        m = re.search('\d\d\_\w*_\d\d\d\d', date)
        if m:
            date = m.group()
        current_output_path = os.path.join(args.outputFolder, date)
        current_processed_file = str(os.path.join(current_output_path, 'processed_urls.txt'))
        try:
            processed_pages = []
            with open(current_processed_file, 'r') as fpro:
                for line in fpro:
                    processed_pages.append(line.strip())
        except:
            processed_pages=[]

        print(processed_pages)
        if not os.path.exists(current_output_path):
            outputPath = Path(current_output_path)
            outputPath.mkdir(parents=True, exist_ok=True)
        all_output_json = []
        current_output_json_file = str(os.path.join(current_output_path, 'summary.json'))
        try:
            with open(current_output_json_file, 'r') as fji:
                all_output_json = json.load(fji)
        except:
            all_output_json = []

        for entry_id, each_entry in enumerate(NewsFeed['entries']):
            current_url = each_entry['link']
            print(each_entry)
            pub_date = each_entry['published']
            if current_url not in processed_pages:
                try:
                    saved_html = os.path.join(current_output_path,str(entry_id+len(processed_pages))+'.html')
                    saved_txt = os.path.join(current_output_path,str(entry_id+len(processed_pages))+'.txt')
                    current_output_json = disease_manager.apply_single_url(current_url, save_txt=saved_txt, date=pub_date)
                    if len(current_output_json['items']) == 0:
                        print('apply backup')
                        current_output_json = backup_manager.apply_single_url(current_url, save_txt=saved_txt, date=pub_date)

                    all_output_json.append(current_output_json)
                    processed_pages.append(current_url)
                    with open(current_output_json_file, 'w') as fo:
                        json.dump(all_output_json, fo)
                    with open(current_processed_file, 'w') as fpro:
                        for each_item in processed_pages:
                            fpro.write(each_item+'\n')
                except Exception as inst:
                    print(inst)

        with open(current_output_json_file, 'w') as fo:
            json.dump(all_output_json, fo)
        with open(current_processed_file, 'w') as fpro:
            for each_item in processed_pages:
                fpro.write(each_item+'\n')



