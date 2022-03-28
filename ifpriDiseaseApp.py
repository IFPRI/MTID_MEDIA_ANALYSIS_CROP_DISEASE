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

class DummyReader:
    def __init__(self, sample, readerPostProcessor):
        self.sample = sample
        self.readerPostProcessor = readerPostProcessor

    def _reset_iter(self):
        pass
    def __iter__(self):
        return self
    def __next__(self):
        return readerPostProcessor.postProcess4Model(self.sample)
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




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputFolder", help="input folder")
    parser.add_argument("--model", help="path to trained model")
    parser.add_argument("--gpu", help="use gpu", default=False, action='store_true')
    parser.add_argument("--config", help="config file")
    parser.add_argument("--outputJson", help="json output", default='diseaseOutput.json')
    parser.add_argument("--outputTsv", help="disease type tsv output", default='diseaseOutput.tsv')
    parser.add_argument("--outputSummary", help="summary output", default='diseaseSummaryOutput.tsv')
    args = parser.parse_args()

    config = ConfigObj(args.config)

    readerPostProcessor = DataPostProcessor(['text'], 'label', config=config)
    mm = ModelManager(gpu=args.gpu, config=config)
    mm.load_model(args.model)

    gs = GateWorker()
    gs_app = GateWorkerAnnotator('gateapp/IFPRI_Diease.xgapp', gs)
    all_input_list = glob.glob(os.path.join(args.inputFolder, '*'))

    fo = open(args.outputTsv, 'w')
    tsv_line = 'file\tstart\tend\tlabel\tsentence\n'
    fo.write(tsv_line)

    doc_counts = {}
    for each_file in all_input_list:
        print(each_file)
        doc_counts[each_file] = {}
        doc_counts[each_file]['damage_type'] = {}

        gdoc = gs.loadDocumentFromFile(each_file)
        pdoc = gs.gdoc2pdoc(gdoc)
        doc_text = pdoc.text
        process_doc = gs_app(pdoc)
        all_anns = process_doc.annset()
        sentence_anns = all_anns.with_type("Sentence")
        eppo_animal_anns = all_anns.with_type("EPPO_animals")
        plant_anns = all_anns.with_type("Plant")
        disease = all_anns.with_type("PlantDisease")

        doc_counts[each_file]['plants'] = get_ann_counts(plant_anns, doc_text)
        doc_counts[each_file]['disease'] = get_ann_counts(disease, doc_text)
        doc_counts[each_file]['pest'] = get_ann_counts(eppo_animal_anns, doc_text)


        for each_ann in sentence_anns:
            sample = {}
            text = doc_text[each_ann.start:each_ann.end]
            sample['text'] = text
            sample['label'] = None
            dmreader = DummyReader(sample, readerPostProcessor)
            oo = mm.apply(dmreader, batch_size=1)
            #print(oo['all_pred_label_string'])
            dmg_type = oo['all_pred_label_string'][0]
            if dmg_type != 'neg':
                tsv_list = [each_file, str(each_ann.start), str(each_ann.end), dmg_type, text]
                print(tsv_list)
                tsv_line = '\t'.join(tsv_list)
                fo.write(tsv_line+'\n')
                if dmg_type not in doc_counts[each_file]['damage_type']:
                    doc_counts[each_file]['damage_type'][dmg_type] = 0
                doc_counts[each_file]['damage_type'][dmg_type] += 1
        gs.deleteResource(gdoc)
    fo.close()

    csv_keys = []
    for each_file in doc_counts:
        for each_output_type in doc_counts[each_file]:
            for each_item in doc_counts[each_file][each_output_type]:
                csv_keys.append(' ||| '.join([each_output_type, each_item]))

    f_csv = open(args.outputSummary, 'w', newline='')
    spamwriter = csv.writer(f_csv, delimiter='\t',quotechar='"', quoting=csv.QUOTE_MINIMAL)
    csv_keys = list(set(csv_keys))
    print(csv_keys)
    csv_headings = ['file_name']+csv_keys
    spamwriter.writerow(csv_headings)

    for each_file in doc_counts:
        csv_line = [each_file]
        for each_key in csv_keys:
            each_key_tok = each_key.split(' ||| ')
            output_type = each_key_tok[0]
            output_item = each_key_tok[1]
            have_value = False
            if output_type in doc_counts[each_file]:
                if output_item in doc_counts[each_file][output_type]:
                    csv_line.append(doc_counts[each_file][output_type][output_item])
                    have_value = True
            if not have_value:
                csv_line.append(0)
        print(csv_line)
        spamwriter.writerow(csv_line)
    f_csv.close()







