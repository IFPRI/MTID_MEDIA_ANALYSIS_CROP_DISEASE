from .PostprocessorBase import ReaderPostProcessorBase

class BERTRelAttPostProcessor(ReaderPostProcessorBase):
    def __init__(self, query_field, x_fields, y_field, word2id=True, label2id=True, **kwargs):
        super().__init__(**kwargs)
        self.query_field = query_field
        self.y_field = y_field
        self.x_fields = x_fields
        self.max_query_len = 200
        self.max_x_len = 200
        self.word2id = word2id
        self.label2id = label2id
        self.sep_tok = '[SEP]'
        self.sep_id = 102
        self.postProcessMethod = 'postProcess4Model'


    #def postProcess(self, sample):
    #    if self.postProcessMethod == 'postProcess4Model':
    #        return self.postProcess4Model(sample)
    #    elif self.postProcessMethod == 'postProcess4Dict':
    #        return self.postProcess4Dict(sample)
    #    elif self.postProcessMethod == 'postProcess4GATEapply':
    #        return self.postProcess4GATEapply(sample)
    #    else:
    #        return sample



    def postProcess4Model(self, sample):
        split_x = []
        for x_field in self.x_fields:
            current_rawx = self._get_sample(sample, x_field)
            split_x.append(str(current_rawx))
        current_rawx = ' '.join(split_x)
        current_bert_tokenized_x = self.bertTokenizer(current_rawx)
        current_query = self._get_sample(sample, self.query_field)
        current_bert_tokenized_query = self.bertTokenizer(current_query)

        query_id = self.bertWord2id(current_bert_tokenized_query, add_special_tokens=True)
        x_id = self.bertWord2id(current_bert_tokenized_x, add_special_tokens=False)

        y = sample[self.y_field]

        if self.label2id:
            y = self.label2ids(y)

        return [x_id, query_id],y


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








