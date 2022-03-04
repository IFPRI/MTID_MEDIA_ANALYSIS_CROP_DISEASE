from gatenlp.gateworker import GateWorker
from gatenlp.gateworker import GateWorkerAnnotator
from gatenlp import Document
from gatenlp.pam.pampac import Ann, AnnAt, Rule, Pampac, AddAnn, N, Seq, Or
from gatenlp.pam.matcher import FeatureMatcher, IfNot
import glob
import argparse
import os
import json


def create_dict_sample(feature_list, text, label, doc):
    current_label = None
    if label == 'pos':
        if 'pest causing the death of affected plant' in feature_list:
            current_label = 'death'
        elif 'pest causing quantitative production losses' in feature_list:
            current_label = 'quantitative'
        elif 'pest causing qualitative production losses' in feature_list:
            current_label = 'qualitative'
    else:
        current_label = 'neg'
    if current_label:
        current_dict = {}
        current_dict['text'] = text
        current_dict['label'] = current_label
        current_dict['doc'] = doc
        return current_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--trainFolder", help="folder to the training data")
    parser.add_argument("--trainDataSuf", help="training data suffix", default='xml')
    parser.add_argument("--trainJsonOutput", help="output training data json format", default='diseaseType.json')
    args = parser.parse_args()

    gs = GateWorker()
    gs_app = GateWorkerAnnotator('gateapp/IFPRI_TrainConvert.xgapp', gs)
    all_training_file_list = glob.glob(os.path.join(args.trainFolder, '*.'+args.trainDataSuf))
    all_training_list = []
    for each_file in all_training_file_list:
        print(each_file)
        gdoc = gs.loadDocumentFromFile(each_file)
        pdoc = gs.gdoc2pdoc(gdoc)
        #pdoc =  Document.load(each_file, fmt='gatexml')
        doc_text = pdoc.text
        #print(pdoc.annset())
        process_doc = gs_app(pdoc)
        #print(process_doc.annset().type_names)
        all_anns = process_doc.annset()
        anns = all_anns.with_type("SentLevelToI")
        for each_ann in anns:
            current_features = each_ann.features.to_dict()
            current_features_list = current_features['all_labels'].lower().split(' ||| ')
            pos_text = doc_text[each_ann.start:each_ann.end]
            print(pos_text)
            pos_dict = create_dict_sample(current_features_list, pos_text, 'pos', each_file)
            if pos_dict:
                all_training_list.append(pos_dict)
        neg_anns = all_anns.with_type("SentLevelToINegative")
        for each_ann in neg_anns:
            neg_text = doc_text[each_ann.start:each_ann.end]
            letters = sum(c.isalpha() for c in neg_text)
            letter_ratio = letters/len(neg_text)
            if letter_ratio > 0.6:
                neg_dict = create_dict_sample([], neg_text, 'neg', each_file)
                all_training_list.append(neg_dict)
        gs.deleteResource(gdoc)
    with open(args.trainJsonOutput, 'w') as fo:
        json.dump(all_training_list, fo)



