from transformers import BertModel
from .miscLayer import SimpleHidden
import os
import torch.nn.functional as F
import torch
import torch.nn as nn
from .BERT_Simple import BERT_Embedding


class SBERT_NLI(nn.Module):
    def __init__(self, config={}, **kwargs):
        super().__init__()
        self._init_params()
        self._read_config(config)

        self.bert_embedding = BERT_Embedding(config)
        bert_dim = self.bert_embedding.bert_dim
        self.hidden_layer = SimpleHidden(bert_dim*3, self.hidden_dim)
        self.layer_output = torch.nn.Linear(self.hidden_dim, self.n_classes)


    def _init_params(self):
        self.hidden_dim = 300
        self.n_classes = 2
        self.sep_idx = 201

    def _read_config(self, config):
        if 'MODEL' in config:
            if 'hidden_dim' in config['MODEL']:
                self.hidden_dim = int(config['MODEL'].get('hidden_dim'))
            if 'n_classes' in config['MODEL']:
                self.n_classes = int(config['MODEL'].get('n_classes'))
            if 'sep_idx' in config['MODEL']:
                self.sep_idx = int(config['MODEL'].get('sep_idx'))


    def forward(self, batchItem, mask=None):
        question = batchItem[2]
        x = batchItem[0]

        question_rep = self.bert_embedding(question, None)
        question_rep_cls = question_rep[0][:,0]

        text_rep = self.bert_embedding(x, None)
        text_rep = text_rep[0]
        text_rep_cls = text_rep[:,0]

        c_hidden = torch.cat([question_rep_cls, text_rep_cls, abs(question_rep_cls - text_rep_cls)], dim=-1)

        hidden = self.hidden_layer(c_hidden)
        out = self.layer_output(hidden)

        y = {
            'y_hat':out,
            'cls_att':out
        }

        return y

