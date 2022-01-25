from .PostprocessorBase import ReaderPostProcessorBase
from transformers import AutoTokenizer

class AutoModelPostProcessor(ReaderPostProcessorBase):
    def __init__(self, x_fields, y_field, query_field=None, word2id=True, label2id=True, **kwargs):
        super().__init__(**kwargs)
        self.query_field = query_field
        self.y_field = y_field
        self.x_fields = x_fields
        self.max_query_len = 200
        self.max_x_len = 200
        if 'READER_POSTPROCESSOR' in self.config:
            if 'max_query_len' in self.config['READER_POSTPROCESSOR']:
                self.max_query_len = int(self.config['READER_POSTPROCESSOR']['max_query_len'])
            if 'max_sent_len' in self.config['READER_POSTPROCESSOR']:
                self.max_x_len = int(self.config['READER_POSTPROCESSOR']['max_sent_len'])
        self.word2id = word2id
        self.label2id = label2id
        self.sep_tok = '[SEP]'
        self.cls_tok = '[CLS]'
        self.sep_id = 102
        self.postProcessMethod = 'postProcess4Model'
        model_name = self.config['BERT']['bert_path']
        self.auto_tokenizer = AutoTokenizer.from_pretrained(model_name)



    def postProcess4Model(self, sample):
        split_x = []
        for x_field in self.x_fields:
            current_rawx = self._get_sample(sample, x_field)
            split_x.append(str(current_rawx))
        current_rawx = ' '.join(split_x)
        if self.query_field:
            current_query = self._get_sample(sample, self.query_field)
            x = self.auto_tokenizer([current_query],[current_rawx],  truncation=True, return_tensors="pt", max_length=self.max_sent_len, padding='max_length')
        else:
            x = self.auto_tokenizer([current_rawx],  truncation=True, return_tensors="pt", max_length=self.max_sent_len, padding='max_length')

        x = x['input_ids'][0].numpy().tolist()
        #print(x)

        y = sample[self.y_field]

        if self.label2id:
            y = self.label2ids(y)

        return x,y


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








