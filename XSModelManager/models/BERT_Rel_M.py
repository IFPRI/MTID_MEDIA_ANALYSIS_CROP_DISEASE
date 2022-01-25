from transformers import BertModel
from .miscLayer import SimpleHidden
import os
import torch.nn.functional as F
import torch
import torch.nn as nn
import math
from .BERT_Simple import BERT_Embedding


class BERT_Rel_M(nn.Module):
    def __init__(self, config={}, **kwargs):
        super().__init__()
        self._init_params()
        self._read_config(config)

        self.bert_embedding = BERT_Embedding(config)
        bert_dim = self.bert_embedding.bert_dim
        self.text_hidden_layer = SimpleHidden(bert_dim, self.hidden_dim)

        #meta_hidden_dim = math.ceil(self.meta_feature_dim/2)
        self.meta_hidden_layer = SimpleHidden(self.meta_feature_dim, self.meta_feature_dim)

        self.hidden_layer = SimpleHidden(self.hidden_dim+self.meta_feature_dim, math.ceil(self.hidden_dim/2))

        self.layer_output = torch.nn.Linear(math.ceil(self.hidden_dim/2), self.n_classes)


    def _init_params(self):
        self.hidden_dim = 50
        self.n_classes = 2
        self.sep_idx = 201
        self.meta_feature_dim = 4

    def _read_config(self, config):
        if 'MODEL' in config:
            if 'hidden_dim' in config['MODEL']:
                self.hidden_dim = int(config['MODEL'].get('hidden_dim'))
            if 'n_classes' in config['MODEL']:
                self.n_classes = int(config['MODEL'].get('n_classes'))
            if 'sep_idx' in config['MODEL']:
                self.sep_idx = int(config['MODEL'].get('sep_idx'))
            if 'meta_feature_dim' in config['MODEL']:
                self.meta_feature_dim = int(config['MODEL'].get('meta_feature_dim'))


    def forward(self, batchItem, mask=None):
        x = batchItem[0]
        meta_matrix = batchItem[2]

        bert_rep = self.bert_embedding(x, mask)
        cls_att = torch.sum(bert_rep[2][11][:,:,0], 1)
        bert_rep = bert_rep[0]
        bert_rep = bert_rep[:,0]

        text_hidden = self.text_hidden_layer(bert_rep)
        meta_hidden = self.meta_hidden_layer(meta_matrix)

        c_hidden = torch.cat((text_hidden,meta_hidden), dim=-1)

        hidden = self.hidden_layer(c_hidden)

        out = self.layer_output(hidden)

        y = {
            'y_hat':out,
            'cls_att':cls_att
        }

        return y

