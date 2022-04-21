from .PostprocessorBase import ReaderPostProcessorBase

class IfpriTrainPostProcessor(ReaderPostProcessorBase):
    def __init__(self, x_fields=['text'], y_field='event', word2id=True, label2id=True, **kwargs):
        super().__init__(**kwargs)
        self.y_field = y_field
        self.x_fields = x_fields
        self.word2id = word2id
        self.label2id = label2id


    def postProcess(self, sample):
        split_x = []
        for x_field in self.x_fields:
            current_rawx = self._get_sample(sample, x_field)
            split_x.append(current_rawx)
        current_rawx = ' '.join(split_x)
        current_cleand_rawx = self.scholarTextClean(current_rawx, lower=True)
        current_bert_tokenized = self.bertTokenizer(current_cleand_rawx)
        current_bert_ided = self.bertWord2id(current_bert_tokenized)

        x = current_bert_ided
        y = sample[self.y_field]

        if self.label2id:
            y = self.label2ids(y)

        return x,y








