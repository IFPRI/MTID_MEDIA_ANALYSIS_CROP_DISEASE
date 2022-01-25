from .PostprocessorBase import ReaderPostProcessorBase
import urllib
import urllib.parse
import re

class SBDMisInfoPostProcessor(ReaderPostProcessorBase):
    def __init__(self, x_fields, y_field, word2id=True, label2id=True, **kwargs):
        super().__init__(**kwargs)

        self.y_field = y_field
        self.x_fields = x_fields
        self.word2id = word2id
        self.label2id = label2id
        self.hashtag_dict = None

    def doc2bow(self, tokened, dict2use):
        gensim_dict = eval(dict2use)
        return gensim_dict.doc2bow(tokened)

    def doc2countHot(self, tokened, dict2use):
        gensim_dict = eval(dict2use)

        gensim_bow_doc = self.doc2bow(tokened, dict2use)
        num_vocab = len(gensim_dict)
        doc_vec = [0] * num_vocab
        for item in gensim_bow_doc:
            vocab_idx = item[0]
            vovab_counts = item[1]
            doc_vec[vocab_idx] = vovab_counts
        return doc_vec

    def postProcess4Model(self, sample):
        split_x = []
        for x_field in self.x_fields:
            current_rawx = self._get_sample(sample, x_field)
            try:
                if len(current_rawx) > 0:
                    split_x.append(current_rawx)
            except:
                pass


        current_rawx = ' '.join(split_x)
        all_hashtags = re.findall('#\S*',current_rawx)
        if self.hashtag_dict:
            current_hash_matrix = self.doc2countHot(all_hashtags, 'self.hashtag_dict')
        else:
            current_hash_matrix = None

        #print(sum(current_hash_matrix))

        #print(current_rawx)
        #print(all_hashtags)
        current_cleand_rawx = self.scholarTextClean(current_rawx, lower=True, strip_html=True, keep_at_mentions=False)
        current_nltk_tokened_rawx = self.nltkTokenizer(current_cleand_rawx)
        current_nltk_tokened_rawx = self.scholarTokenClean(current_nltk_tokened_rawx, stopwords=self.stop_words)
        current_nltk_tokened_rawx = [t for t in current_nltk_tokened_rawx if t != '_']
        #print(current_nltk_tokened_rawx)
        if self.gensim_dict:
            current_count_matrix = self.doc2countHot(current_nltk_tokened_rawx, 'self.gensim_dict')
        else:
            current_count_matrix = None

        current_bert_tokenized = self.bertTokenizer(current_cleand_rawx)
        current_bert_ided = self.bertWord2id(current_bert_tokenized)
        y = sample[self.y_field]
        if self.label2id:
            y = self.label2ids(y)


        inclided_urls = self._get_sample(sample, 'extended_url')
        ###to be remove
        #inclided_urls = [inclided_urls]
        #base_url_list = self._get_base_url_list(inclided_urls)
        #if len(base_url_list) > 0:
        #    base_url = base_url_list[0]
        #else:
        #    base_url = ''




        output_dict = {}
        output_dict['bert_tokenized'] = current_bert_tokenized
        output_dict['bert_ided'] = current_bert_ided
        output_dict['count_matrix'] = current_count_matrix
        output_dict['nltk_tokened_rawx'] = current_nltk_tokened_rawx
        output_dict['hash_matrix'] = current_hash_matrix
        output_dict['target_label'] = y
        return output_dict

    def _get_base_url_list(self, inclided_urls):
        base_url_list = []
        for each_url in inclided_urls:
            if each_url:
                #print(each_url)
                parsed_url = urllib.parse.urlparse(each_url)
                netloc = parsed_url.netloc
                #print(parsed_url)
                #print(netloc)
                if netloc:
                    netloc_tok = netloc.split('.')
                    if netloc_tok[0] == 'www':
                        netloc = '.'.join(netloc_tok[1:])
                    else:
                        netloc = '.'.join(netloc_tok)
                    #print(each_url)
                    #print(netloc)
                    base_url_list.append(netloc)

        return base_url_list



    def postProcess4hashTagDict(self, sample):
        split_x = []
        for x_field in self.x_fields:
            current_rawx = self._get_sample(sample, x_field)
            try:
                if len(current_rawx) > 0:
                    split_x.append(current_rawx)
            except:
                pass


        current_rawx = ' '.join(split_x)
        all_hashtags = re.findall('#\S*',current_rawx)
        return all_hashtags


    def postProcess4Pretrain(self, sample):
        split_x = []
        for x_field in self.x_fields:
            current_rawx = self._get_sample(sample, x_field)
            split_x.append(current_rawx)

        current_rawx = ' '.join(split_x)
        current_cleand_rawx = self.scholarTextClean(current_rawx, lower=True, strip_html=True, keep_at_mentions=False)
        current_nltk_tokened_rawx = self.nltkTokenizer(current_cleand_rawx)
        current_nltk_tokened_rawx = self.scholarTokenClean(current_nltk_tokened_rawx, stopwords=self.stop_words)
        current_nltk_tokened_rawx = [t for t in current_nltk_tokened_rawx if t != '_']
        if self.gensim_dict:
            current_count_matrix = self.doc2countHot(current_nltk_tokened_rawx)
        else:
            current_count_matrix = None

        current_bert_tokenized = self.bertTokenizer(current_cleand_rawx)
        current_bert_ided = self.bertWord2id(current_bert_tokenized)

        output_dict = {}
        output_dict['bert_tokenized'] = current_bert_tokenized
        output_dict['bert_ided'] = current_bert_ided
        output_dict['count_matrix'] = current_count_matrix
        output_dict['nltk_tokened_rawx'] = current_nltk_tokened_rawx
        output_dict['target_label'] = 0
        return output_dict



























