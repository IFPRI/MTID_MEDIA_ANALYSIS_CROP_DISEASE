import sys
import argparse
import os
from pathlib import Path
script_path = os.path.abspath(__file__)
dir_path = os.path.dirname(script_path)
sys.path.insert(0, str(dir_path))
#print(dir_path)
#from XSModelManager.ModelManager import ModelManager
#from XSModelManager.ModelManager_CANTM import ModelManager_CANTM as ModelManager
global logging
import logging
from configobj import ConfigObj
import copy
from sklearn.model_selection import KFold
import random
from gensim.corpora.dictionary import Dictionary
#global ModelManager
#global DataPostProcessor
#global readerPostProcessor

class CrossValidator:
    def __init__(self):
        self._reset_iter()
        self.n_folds = 0
        self.speaker_id_dict = {}

    def reconstruct_ids(self, each_fold):
        output_ids = [[],[]] #[train_ids, test_ids]
        for sp_id in range(len(each_fold)):
            current_output_ids = output_ids[sp_id]
            current_fold_ids = each_fold[sp_id]
            for doc_id in current_fold_ids:
                current_output_ids.append(self.all_ids[doc_id])
        return output_ids

    def reconstruct_ids_speaker(self, each_fold):
        output_ids = [[],[]] #[train_ids, test_ids]
        for sp_id in range(len(each_fold)):
            current_output_ids = output_ids[sp_id]
            current_fold_ids = each_fold[sp_id]
            for doc_id in current_fold_ids:
                current_output_ids += self.speaker_id_dict[self.speaker_id_dict_keys[doc_id]]
        return output_ids

    def cross_validation(self, dataIter, n_folds=5, speaker_field=None):
        self._reset_iter()
        if speaker_field:
            self.leave_speaker_out = True
        else:
            self.leave_speaker_out = False
        self.dataIter = copy.deepcopy(dataIter)
        self.valdataIter = copy.deepcopy(dataIter)
        self.valdataIter.shuffle = False
        self.n_folds = n_folds
        kf = KFold(n_splits=self.n_folds)
        self.all_ids = copy.deepcopy(dataIter.all_ids)

        if self.leave_speaker_out:
            self.speaker_id_dict = self.get_all_speaker_ids(speaker_field)
            self.speaker_id_dict_keys = list(self.speaker_id_dict.keys())
            print(self.speaker_id_dict_keys)
            random.shuffle(self.speaker_id_dict_keys)
            self.kfIter = kf.split(self.speaker_id_dict_keys)
        else:
            random.shuffle(self.all_ids)
            self.kfIter = kf.split(self.all_ids)

    def __len__(self):
        return self.n_folds

    def get_all_speaker_ids(self, speaker_field):
        speaker_id_dict = {}
        for item in self.dataIter:
            current_id = self.dataIter.current_sample_dict_id
            current_speaker_id = self.dataIter.data_dict[current_id][speaker_field]

            if current_speaker_id not in speaker_id_dict:
                speaker_id_dict[current_speaker_id] = [current_id]
            else:
                speaker_id_dict[current_speaker_id].append(current_id)

        return speaker_id_dict

    def __iter__(self):
        self._reset_iter()
        return self

    def __next__(self):
        if self.current_fold < self.n_folds:
            self.current_fold += 1
            if self.leave_speaker_out:
                train_ids, test_ids = self.reconstruct_ids_speaker(next(self.kfIter))
            else:
                train_ids, test_ids = self.reconstruct_ids(next(self.kfIter))
            #print(train_ids, test_ids)
            self.dataIter.all_ids = copy.deepcopy(train_ids)
            self.valdataIter.all_ids = copy.deepcopy(test_ids)
            return self.dataIter, self.valdataIter

        else:
            self._reset_iter()
            raise StopIteration

    def _reset_iter(self):
        self.current_fold = 0



class ExpManager:
    def __init__(self, dictargs):
        self.dictargs = dictargs
        self.modelType = dictargs.get('modelType')
        if dictargs.get('cuda_device'):
            os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_device

        if dictargs.get('debug'):
            logging.basicConfig(level=logging.DEBUG)
        else:
            logging.basicConfig(level=logging.INFO)

        self.config = {}
        if dictargs.get('configFile'):
            self.config = ConfigObj(dictargs.get('configFile'))

        self.updateTarget = not dictargs.get('noUpdateTarget')
        self._loadDataReader(dictargs.get('readerType'))
        self._loadModelManager(dictargs)
        #self.early_stop_function = train_loss_early_stopping

    def _loadDataReader(self, readerType):
        global DataReader
        if readerType == 'tsv':
            from XSNLPReader import TSVReader as DataReader
        elif readerType == 'json':
            from XSNLPReader import JSONReader as DataReader

    def _loadModelManager(self, dictargs):
        self._stage_controller('loadModelManager')

    def train(self):
        self._train_perpare()
        if self.dictargs.get('testInput'):
            self._test_perpare()
            val_result = self.trainModel(self.train_dataIter, self.test_dataIter, self.dictargs.get('savePath'))
        else:
            val_result = self.trainModel(self.train_dataIter, None, self.dictargs.get('savePath'))

        print(val_result)

    def show_topics(self):
        mm = ModelManager(gpu=dictargs.get('gpu'), config=self.config)
        if self.dictargs.get('loadPath'):
            mm.load_model(self.dictargs.get('loadPath'))
        mm.getTopics()

    def test(self):
        self._test_perpare()
        mm = ModelManager(gpu=dictargs.get('gpu'), config=self.config)
        if self.dictargs.get('loadPath'):
            mm.load_model(self.dictargs.get('loadPath'))
            self.test_dataIter.target_labels = mm.target_labels
            self.test_dataIter.updateTargetLabels2PostProcessor()
        else:
            mm.genPreBuildModel(self.modelType)
            mm.target_labels = self.test_dataIter.target_labels

        #print(next(self.test_dataIter))
        results_dict = mm.eval(self.test_dataIter, batch_size=self.dictargs.get('batch_size'), return_error_list=self.dictargs.get('return_error_list'), return_correct_list=self.dictargs.get('return_correct_list'))
        print(results_dict['accuracy'])
        print(results_dict['f1-avg'])
        print(results_dict['f-measure'])
        print(results_dict['confusion_matrix'])


        if self.dictargs.get('return_error_list'):
            error_list_file_path = os.path.join(self.dictargs.get('savePath'), 'error_list.tsv')
            self.write_analysis_file(error_list_file_path, results_dict['errorSample'])

        if self.dictargs.get('return_correct_list'):
            correct_list_file_path = os.path.join(self.dictargs.get('savePath'), 'correct_list.tsv')
            self.write_analysis_file(correct_list_file_path, results_dict['correctSample'])

    def write_analysis_file(self, file_path, sample_list):
        with open(file_path, 'w') as fo:
            head_line = 'prediction\tlabel\ttext'
            if dictargs.get('addi_err_analysis_fields'):
                for each_addi_field in dictargs.get('addi_err_analysis_fields').split(','):
                    head_line += '\t' + each_addi_field
            fo.write(head_line+'\n')
            for analysis_line in sample_list:
                fo.write(analysis_line.strip()+'\n')


 
    def cross_validation(self):
        self._train_perpare()
        corss_validator = CrossValidator()
        corss_validator.cross_validation(self.train_dataIter, n_folds=self.dictargs.get('nFold'), speaker_field=self.dictargs.get('leave_speaker'))
        current_fold = 0
        results_dict = {}
        results_dict['accuracy'] = []
        results_dict['f1-avg'] = []
        results_dict['perplexity'] = []
        results_dict['log_perplexity'] = []
        results_dict['perplexity_x_only'] = []
        results_dict['f-measure'] = {}
        results_dict['confusion_matrix'] = {}
        results_dict['errorSample'] = []
        #results_dict['pred_list'] = []
        #all_acc = 0
        #all_f1 = 0

        for train_dataIter, val_dataIter in corss_validator:
            infomessage = 'Training Fold '+str(current_fold)
            logging.info(infomessage)
            fold_folder = 'fold_'+str(current_fold)
            save_path = os.path.join(self.dictargs.get('savePath'), fold_folder)
            val_result = self.trainModel(train_dataIter, val_dataIter, save_path, earlyStoppingFunction=train_loss_early_stopping)
            results_dict = self.merge_folds_results(results_dict, val_result)
            current_fold += 1

        #print(results_dict)
        output_metrics = ['precision', 'recall', 'f1']
        for metric in output_metrics:
            print(metric)
            print(self.get_f1_results(results_dict, metric))

        print('confusion matrix')
        print(self.get_cm_results(results_dict))

        avg_f1 = sum(results_dict['f1-avg'])/len(results_dict['f1-avg'])
        avg_acc = sum(results_dict['accuracy'])/len(results_dict['accuracy'])

        if self.dictargs.get('return_error_list'):
            error_list_file_path = os.path.join(self.dictargs.get('savePath'), 'error_list.tsv')
            with open(error_list_file_path, 'w') as fo:
                head_line = 'prediction\tlabel\ttext'
                if dictargs.get('addi_err_analysis_fields'):
                    for each_addi_field in dictargs.get('addi_err_analysis_fields').split(','):
                        head_line += '\t' + each_addi_field

                fo.write(head_line+'\n')
                
                for each_error_line in results_dict['errorSample']:
                    fo.write(each_error_line.strip()+'\n')


        print('accuracy', avg_acc, 'f1', avg_f1)

    def merge_folds_results(self, results_dict, val_result):
        results_dict['accuracy'].append(val_result['accuracy'])
        results_dict['f1-avg'].append(val_result['f1-avg'])
        results_dict = self.merge_class_dict(results_dict, val_result, 'f-measure')
        results_dict = self.merge_class_dict(results_dict, val_result, 'confusion_matrix')
        if self.dictargs.get('return_error_list'):
            results_dict['errorSample'] += val_result['errorSample']

        #    results_dict['error_list'] +=  val_result['error_list']
        #    results_dict['pred_list'] +=  val_result['pred_list']

        return results_dict


    def merge_class_dict(self, results_dict, val_result, results_fields):
        if results_fields in val_result:
            for each_class in val_result[results_fields]:
                if each_class not in results_dict[results_fields]:
                    results_dict[results_fields][each_class] = {}
                for metrics in val_result[results_fields][each_class]:
                    if metrics not in results_dict[results_fields][each_class]:
                        results_dict[results_fields][each_class][metrics] = []
                    results_dict[results_fields][each_class][metrics].append(val_result[results_fields][each_class][metrics])
        return results_dict


    def get_f1_results(self, results_dict, field):
        class_avg_score = {}
        for class_field in results_dict['f-measure']:
            #print(results_dict['f-measure'][class_field])
            score = sum(results_dict['f-measure'][class_field][field])
            t = len(results_dict['f-measure'][class_field][field])
            class_avg_score[class_field] = score/t
        return class_avg_score


    def get_cm_results(self, results_dict):
        class_avg_score = {}
        for class_field in results_dict['confusion_matrix']:
            class_avg_score[class_field] = {}
            for each_class in results_dict['confusion_matrix'][class_field]:
                score = sum(results_dict['confusion_matrix'][class_field][each_class])
                class_avg_score[class_field][each_class] = score
        return class_avg_score


    def trainModel(self, train_dataIter, val_dataIter, save_path, earlyStoppingFunction=None):
        if self.dictargs.get('arguementedData'):
            train_dataIter._read_file(self.dictargs.get('arguementedData'))
        if self.dictargs.get('label_transfer'):
            train_dataIter.transferLabel()
            if val_dataIter:
                val_dataIter.transferLabel()
        if self.dictargs.get('label_remove'):
            train_dataIter.removeLabel()
            if val_dataIter:
                val_dataIter.removeLabel()


        dummy_config, train_dataIter, val_dataIter = self._stage_controller('trainModel', train_dataIter, val_dataIter)

        if dictargs.get('calSampleWeight'):
            train_dataIter.cal_sample_weights()
            sample_weights = train_dataIter.label_weights_list
        else:
            sample_weights = None


        dummy_config['MODEL'].update({'n_classes':len(train_dataIter.target_labels), 'sample_weights':sample_weights})
        if 'MODEL' in self.config:
            self.config['MODEL'].update(dummy_config['MODEL'])
        else:
            self.config['MODEL'] = dummy_config['MODEL']

        print(self.config)

        mm = ModelManager(gpu=dictargs.get('gpu'), config=self.config)
        if self.dictargs.get('loadPath'):
            mm.load_model(self.dictargs.get('loadPath'))
        else:
            mm.genPreBuildModel(self.modelType)

        mm.train(train_dataIter, save_path=save_path, valDataIter=val_dataIter, earlyStopping=True, patience=self.dictargs.get('patience'), batch_size=self.dictargs.get('batch_size'), warm_up=self.dictargs.get('warm_up'), earlyStoppingFunction=earlyStoppingFunction, num_epoches=self.dictargs.get('num_epoches'))
        if val_dataIter:
            result_dict = mm.eval(val_dataIter, batch_size=self.dictargs.get('batch_size'), return_error_list=self.dictargs.get('return_error_list'))
        else:
            result_dict = {}
        return result_dict




    def _train_perpare(self):
        if not self.dictargs.get('trainInput'):
            print('trainInput is required')
            sys.exit()
        self.train_dataIter = DataReader(self.dictargs.get('trainInput'), postProcessor=self.readerPostProcessor, updateTarget=self.updateTarget, config=self.config, shuffle=True)
        if self.dictargs.get('label_transfer'):
            self.train_dataIter.transferLabel()
        if self.dictargs.get('label_remove'):
            self.train_dataIter.removeLabel()

    def _test_perpare(self):
        self.test_dataIter = DataReader(self.dictargs.get('testInput'), postProcessor=self.readerPostProcessor, updateTarget=self.updateTarget, config=self.config)
        if self.dictargs.get('label_remove'):
            self.test_dataIter.removeLabel()
        if self.dictargs.get('label_transfer'):
            self.test_dataIter.transferLabel()


    def _stage_controller(self, stage, *args):
        if stage == 'loadModelManager':
            global ModelManager
            global DataPostProcessor
            x_fields = dictargs.get('x_fields').split(',')
            if dictargs.get('addi_err_analysis_fields'):
                addi_err_fields = dictargs.get('addi_err_analysis_fields').split(',')
            else:
                addi_err_fields = None

            if self.modelType == 'CANTM':
                from XSModelManager.ModelManager_CANTM import ModelManager_CANTM as ModelManager
                from XSNLPReader.readerPostProcessor import CANTMpostProcessor as DataPostProcessor
                self.readerPostProcessor = DataPostProcessor(x_fields=x_fields, y_field=dictargs.get('y_field'), config=self.config, additional_raw_fields=addi_err_fields)
            elif self.modelType == 'SBDFC':
                from XSModelManager.ModelManager_CANTM import ModelManager_CANTM as ModelManager
                from XSNLPReader.readerPostProcessor import SBDMisInfoPostProcessor as DataPostProcessor
                self.readerPostProcessor = DataPostProcessor(x_fields=x_fields, y_field=dictargs.get('y_field'), config=self.config, additional_raw_fields=addi_err_fields)
            elif self.modelType == 'BERT_Rel_M':
                from XSModelManager.ModelManager import ModelManager
                from XSNLPReader.readerPostProcessor import SBDSentRelPostProcessor as DataPostProcessor
                self.readerPostProcessor = DataPostProcessor(dictargs.get('query_field'), x_fields, dictargs.get('y_field'), config=self.config, additional_raw_fields=addi_err_fields)
            elif self.modelType == 'CANTM_Pretrain':
                from XSModelManager.ModelManager_CANTM import ModelManager_CANTMPreTrain as ModelManager
                from XSNLPReader.readerPostProcessor import CANTMpostProcessor as DataPostProcessor
                self.readerPostProcessor = DataPostProcessor(x_fields=x_fields, y_field=dictargs.get('y_field'), config=self.config, additional_raw_fields=addi_err_fields)
                self.readerPostProcessor.postProcessMethod = 'postProcess4Pretrain'
                self.updateTarget = False
            elif self.modelType == 'BERT_Rel_Att':
                from XSModelManager.ModelManager import ModelManager
                from XSNLPReader.readerPostProcessor import BERTRelAttPostProcessor as DataPostProcessor
                self.readerPostProcessor = DataPostProcessor(dictargs.get('query_field'), x_fields, dictargs.get('y_field'), config=self.config, additional_raw_fields=addi_err_fields)
            elif self.modelType == 'BERT_Rel_Simple':
                from XSModelManager.ModelManager import ModelManager
                from XSNLPReader.readerPostProcessor import SentRelPostProcessor as DataPostProcessor
                self.readerPostProcessor = DataPostProcessor(dictargs.get('query_field'), x_fields, dictargs.get('y_field'), config=self.config, additional_raw_fields=addi_err_fields)
            elif self.modelType == 'BERT_Simple':
                from XSModelManager.ModelManager import ModelManager
                from XSNLPReader.readerPostProcessor import BertPostProcessor as DataPostProcessor
                self.readerPostProcessor = DataPostProcessor(x_fields, self.dictargs.get('y_field'), config=self.config, additional_raw_fields=addi_err_fields)
            elif self.modelType == 'SBERT_NLI':
                from XSModelManager.ModelManager import ModelManager
                from XSNLPReader.readerPostProcessor import SBERTNLIPostProcessor as DataPostProcessor
                self.readerPostProcessor = DataPostProcessor(dictargs.get('query_field'), x_fields, dictargs.get('y_field'), config=self.config, additional_raw_fields=addi_err_fields)
            elif self.modelType == 'AutoModel':
                from XSModelManager.ModelManager_AutoModel import ModelManager_AutoModel as ModelManager
                from XSNLPReader.readerPostProcessor import AutoModelPostProcessor as DataPostProcessor
                self.readerPostProcessor = DataPostProcessor(x_fields, self.dictargs.get('y_field'), config=self.config, additional_raw_fields=addi_err_fields)
            elif self.modelType == 'AutoModelSeqClass':
                from XSModelManager.ModelManager import ModelManager
                from XSNLPReader.readerPostProcessor import AutoModelPostProcessor as DataPostProcessor
                self.readerPostProcessor = DataPostProcessor(x_fields, self.dictargs.get('y_field'), config=self.config, additional_raw_fields=addi_err_fields)

            else:
                print('model type not supported, supported model types are BERT_Simple, CANTM, BERT_Rel_Att, BERT_Rel_Simple')
                sys.exit()
        if stage == 'trainModel':
            #print(*args)
            dummy_config = {}
            dummy_config['MODEL'] = {}
            train_dataIter = args[0]
            val_dataIter = args[1]
            if self.modelType == 'SBDFC':
                train_dataIter.postProcessor.hashtag_dict = train_dataIter.dictBuild('postProcess4hashTagDict')
                hash_tag_dict_dim = len(train_dataIter.postProcessor.hashtag_dict)
                dummy_config['MODEL']['hash_tag_dim'] = hash_tag_dict_dim
                if val_dataIter:
                    val_dataIter.postProcessor.hashtag_dict = train_dataIter.postProcessor.hashtag_dict

            if self.modelType == 'BERT_Rel_M':
                dummy_config['MODEL']['meta_feature_dim'] = train_dataIter.postProcessor.num_meta_feature


            if self.modelType == 'CANTM' or self.modelType == 'CANTM_Pretrain' or self.modelType == 'SBDFC':
                if self.dictargs.get('loadPath'):
                    dict_path = os.path.join(self.dictargs.get('loadPath'), 'gensim_dict.pt')
                    train_dataIter.loadDict(dict_path)
                else:
                    train_dataIter.buildDict()
                if val_dataIter:
                    val_dataIter.updateDict(train_dataIter.postProcessor.gensim_dict)
                vocab_dim = len(train_dataIter.postProcessor.gensim_dict)
            else:
                vocab_dim = 0

            dummy_config['MODEL']['vocab_dim'] = vocab_dim


            return dummy_config, train_dataIter, val_dataIter


def train_loss_early_stopping(train_ouput, eval_output, best_saved):
        if best_saved == 0:
            best_saved = 999
        score2compare = train_ouput['loss']
        if score2compare < best_saved:
            return score2compare, False
        else:
            return best_saved, True




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--trainInput", help="training file input path")
    parser.add_argument("--arguementedData", help="arguemented data for training")
    parser.add_argument("--testInput", help="testing file input path")
    parser.add_argument("--readerType", help="supported readerType: tsv, json", default='json')
    parser.add_argument("--splitValidation", type=float, help="split data from training for validation")
    parser.add_argument("--nFold", type=int, help="n fold crossvalidation")
    parser.add_argument("--savePath", default='.',help="model save path")
    parser.add_argument("--configFile", help="config files if needed")
    parser.add_argument("--x_fields", help="x fileds", default='text')
    parser.add_argument("--y_field", help="y filed", default=None)
    parser.add_argument("--query_field", help="query fileds", default='query')
    parser.add_argument("--noUpdateTarget", help="not update targets when reading the training input", default=False, action='store_true')
    parser.add_argument("--batch_size", type=int, default=16, help="batch size")
    parser.add_argument("--gpu", help="use gpu", default=False, action='store_true')
    parser.add_argument("--num_epoches", type=int, default=200, help="number of training epoches")
    parser.add_argument("--warm_up", type=int, default=5, help="warm up epoches")
    parser.add_argument("--patience", type=int, default=5, help="early stopping patience")
    parser.add_argument("--leave_speaker", help="leave speaker out for cross validation")
    parser.add_argument("--modelType", help="supported readerType: CANTM, BERT_Simple, BERT_Rel_Simple, BERT_Rel_Att, BERT_Rel_M, AutoModel, AutoModelSeqClass", default='CANTM')
    parser.add_argument("--debug", help="debug mode", default=False, action='store_true')
    parser.add_argument("--calSampleWeight", help="get sample weight for unbanlanced sample", default=False, action='store_true')
    parser.add_argument("--label_remove", help="remove label according to config", default=False, action='store_true')
    parser.add_argument("--label_transfer", help="transfer label according to config", default=False, action='store_true')
    parser.add_argument("--loadPath", help="model load path")
    parser.add_argument("--cuda_device", help="set cuda visible device")
    parser.add_argument("--return_error_list", help="return error list", default=False, action='store_true')
    parser.add_argument("--return_correct_list", help="return correct list", default=False, action='store_true')
    parser.add_argument("--addi_err_analysis_fields", help="return addition fileds in error list")
    parser.add_argument("--show_topics", help="show topics", default=False, action='store_true')
    args = parser.parse_args()

    dictargs = vars(args)
    print(dictargs)
    expManager = ExpManager(dictargs)
    if args.nFold:
        expManager.cross_validation()
    elif args.splitValidation:
        pass
    elif args.trainInput:
        expManager.train()
    elif args.testInput:
        expManager.test()

    if args.show_topics:
        expManager.show_topics()









