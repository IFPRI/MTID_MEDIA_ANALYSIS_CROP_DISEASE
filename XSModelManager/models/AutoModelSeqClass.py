#from transformers import BertModel
from transformers import AutoModelForSequenceClassification
from .miscLayer import SimpleHidden
import os
import torch.nn.functional as F
import torch
import torch.nn as nn


class AutoModelSeqClass(nn.Module):
    def __init__(self, config={}, **kwargs):
        super().__init__()
        self._read_config(config)
        self.atseqcls = AutoModelForSequenceClassification.from_pretrained(self.pretrain_path)
        if 'BERT' in config:
            if 'trainable_layers' in config['BERT']:
                print(config['BERT']['trainable_layers'])
                for name, param in self.atseqcls.named_parameters():
                    if name in config['BERT']['trainable_layers']:
                        param.requires_grad = True
                    else:
                        param.requires_grad = False


    def _read_config(self, config):
        #print(config)
        try:
            self.pretrain_path = config['BERT']['bert_path']
        except:
            print('using default')
            self.pretrain_path = "cross-encoder/stsb-roberta-large"



    def forward(self, batchItem, mask=None):
        x = batchItem[0]
        seqscores = self.atseqcls(x)


        y = {
            'y_hat':seqscores.logits,
            'cls_att':seqscores.logits
        }
        #print(seqscores.logits)
        #print(seqscores)
        
        #print(seqscores.logits.shape)

        return y

