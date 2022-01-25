from .PostprocessorBase import ReaderPostProcessorBase
import os
import glob
import re

class SBDSentRelPostProcessor(ReaderPostProcessorBase):
    def __init__(self, query_field, x_fields, y_field, word2id=True, label2id=True, **kwargs):
        super().__init__(**kwargs)
        self.query_field = query_field
        self.y_field = y_field
        self.x_fields = x_fields
        self.max_query_len = 250
        self.max_x_len = 250
        self.word2id = word2id
        self.label2id = label2id
        self.sep_tok = '[SEP]'
        self.cls_tok = '[CLS]'
        self.sep_id = 102
        self.postProcessMethod = 'postProcess4Model'
        self.readMetaDataFile()


    def readMetaDataFile(self):
        try:
            meta_file_path = self.config['READER_POSTPROCESSOR']['meta_files']
        except:
            meta_file_path = None

        self.one_hot_meta = {}
        if meta_file_path:
            all_meta_files = glob.glob(os.path.join(meta_file_path, '*.txt'))
            for each_meta_file in all_meta_files:
                base_file_name = os.path.basename(each_meta_file)
                base_prefix_tok = base_file_name.split('.')[0].split('_')
                tmp_list = []
                with open(each_meta_file, 'r') as fin:
                    for line in fin:
                        striped_line = line.strip()
                        if len(striped_line) > 0:
                            tmp_list.append(striped_line)
                self.one_hot_meta[base_file_name] = {}
                self.one_hot_meta[base_file_name]['list'] = tmp_list
                self.one_hot_meta[base_file_name]['mode'] = base_prefix_tok[-1]
                self.one_hot_meta[base_file_name]['field'] = base_prefix_tok[-2]
        #print(self.one_hot_meta)
        self.num_meta_feature = len(self.one_hot_meta)+1


    def getMetaOneHot(self, one_hot_search_dict):
        hot_list = [len(one_hot_search_dict.get('text'))/self.max_x_len]
        for feature_key in self.one_hot_meta:
            mode = self.one_hot_meta[feature_key]['mode']
            field = self.one_hot_meta[feature_key]['field']
            if mode == 'search':
                hot_list.append(self._regexList_search(str(one_hot_search_dict.get(field)), self.one_hot_meta[feature_key]['list']))
            if mode == 'match':
                hot_list.append(self._list_match(one_hot_search_dict.get(field), self.one_hot_meta[feature_key]['list']))
        return hot_list


    def _list_match(self, text_list, match_list):
        num_match = 0
        #print(text_list, match_list)
        for each_text in text_list:
            #print(each_text)
            if each_text in match_list:
                num_match += 1
                #print(each_text)
        return num_match



    def _regexList_search(self, text, regexList):
        #print(text)
        match = 0
        for each_rule in regexList:
            m = re.search(each_rule, text)
            if m:
                match = 1
                #print(each_rule,m)
                break
        return match

    def postProcess4Model(self, sample):
        split_x = []
        for x_field in self.x_fields:
            current_rawx = self._get_sample(sample, x_field)
            split_x.append(str(current_rawx))
        current_rawx = ' '.join(split_x)
        all_hashtags = re.findall('#\S*',current_rawx)
        all_hashtags = [ss.lower() for ss in all_hashtags]
        included_urls = self._get_sample(sample, 'extended_url')
        #print(included_urls)
        if included_urls == 'None':
            extended_urls = ''
        else:
            extended_urls = []
            for url_key in included_urls:
                extended_urls.append(included_urls[url_key])
            extended_urls = ' ||| '.join(extended_urls)

        one_hot_search_dict = {
                'text':current_rawx,
                'url':extended_urls,
                'hashtag':all_hashtags,
                }

        hot_list = self.getMetaOneHot(one_hot_search_dict)
        current_query = self._get_sample(sample, self.query_field)
        current_bert_tokenized_query = self.bertTokenizer(current_query)[:self.max_query_len]
        current_bert_tokenized_query.append(self.sep_tok)
        max_x_len = self.max_sent_len - len(current_bert_tokenized_query)
        current_bert_tokenized_x = self.bertTokenizer(current_rawx)[:max_x_len]
        x = self.bertWord2id(current_bert_tokenized_query+current_bert_tokenized_x, add_special_tokens=True)
        #print(x)


        y = sample[self.y_field]

        if self.label2id:
            y = self.label2ids(y)

        output_dict = {}
        output_dict['bert_ided'] = x
        output_dict['meta_matrix'] = hot_list
        output_dict['target_label'] = y


        return output_dict


    def postProcess4GATEapply(self, sample):
        split_x = []
        for x_field in self.x_fields:
            current_rawx = self._get_sample(sample, x_field)
            split_x.append(str(current_rawx))
        current_rawx = ' '.join(split_x)
        current_cleand_rawx = self.scholarTextClean(current_rawx, lower=True)
        current_bert_tokenized = self.bertTokenizer(current_cleand_rawx)

        y = sample[self.y_field]

        if self.label2id:
            y = self.label2ids(y)
        return sample, current_bert_tokenized, y 








