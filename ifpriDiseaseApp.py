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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputFolder", help="input folder")
    parser.add_argument("--model", help="path to trained model")
    parser.add_argument("--gpu", help="use gpu", default=False, action='store_true')
    parser.add_argument("--config", help="config file")
    parser.add_argument("--outputTsv", help="tsv output", default='diseaseOutput.tsv')
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

    for each_file in all_input_list:
        print(each_file)
        gdoc = gs.loadDocumentFromFile(each_file)
        pdoc = gs.gdoc2pdoc(gdoc)
        doc_text = pdoc.text
        process_doc = gs_app(pdoc)
        all_anns = process_doc.annset()
        sentence_anns = all_anns.with_type("Sentence")
        for each_ann in sentence_anns:
            sample = {}
            text = doc_text[each_ann.start:each_ann.end]
            sample['text'] = text
            sample['label'] = None
            dmreader = DummyReader(sample, readerPostProcessor)
            oo = mm.apply(dmreader, batch_size=1)
            #print(oo['all_pred_label_string'])
            if oo['all_pred_label_string'][0] != 'neg':
                tsv_list = [each_file, str(each_ann.start), str(each_ann.end), oo['all_pred_label_string'][0], text]
                print(tsv_list)
                tsv_line = '\t'.join(tsv_list)
                fo.write(tsv_line+'\n')
        gs.deleteResource(gdoc)
    fo.close()



