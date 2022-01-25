from transformers import BertModel
from .miscLayer import SimpleHidden
import os
import torch.nn.functional as F
import torch
import torch.nn as nn
from .BERT_Simple import BERT_Embedding


class QuestionAttention(nn.Module):
    def __init__(self, d_model, d_output, dropout = 0.1):
        super().__init__()

        self.q_linear = nn.Linear(d_model, d_output)
        self.dropout_q = nn.Dropout(dropout)

        self.v_linear = nn.Linear(d_model, d_output)
        self.dropout_v = nn.Dropout(dropout)
        self.k_linear = nn.Linear(d_model, d_output)
        self.dropout_k = nn.Dropout(dropout)
        self.softmax_simi = nn.Softmax(dim=1)

    def forward(self, question, text):

        #q = self.q_linear(question)
        #q = F.relu(q)
        #q = self.dropout_q(q)

        #k = self.k_linear(text)
        #k = F.relu(k)
        #k = self.dropout_k(k)

        #v = self.v_linear(text)
        #v = F.relu(v)
        #v = self.dropout_v(v)
        q=question
        k=text
        v=text
        
        q = q.unsqueeze(1)
        batch_size, output_len, dimensions = q.size()
        query_len = k.size(1)

        #print(q.shape)
        #print(k.shape)

        #q=question
        #k=text
        #v=text

        #dotProducSimi = k.matmul(q.unsqueeze(2))
        #print(k.shape)
        #print(q.shape)
        dotProducSimi = torch.bmm(q, k.transpose(1, 2).contiguous())
        #dotProducSimi = torch.matmul(q, k.transpose(-2, -1))
        dotProducSimi = dotProducSimi.view(batch_size * output_len, query_len)
        #print(dotProducSimi.shape)
        normedSimi = self.softmax_simi(dotProducSimi)
        #print(normedSimi.shape)
        normedSimi = normedSimi.view(batch_size, output_len, query_len)
        #print(normedSimi.shape)

        attVector = torch.bmm(normedSimi, v)
        #print(attVector.shape)

        weightedSum = torch.sum(attVector, dim=1)
        weightedSum = F.relu(self.q_linear(weightedSum))

        #output = self.out(weightedSum)
        #print(weightedSum.shape)
        return weightedSum




class BERT_Rel_Att(nn.Module):
    def __init__(self, config={}, **kwargs):
        super().__init__()
        self._init_params()
        self._read_config(config)

        #self.bert_question_embedding = BERT_Embedding(config)
        self.bert_text_embedding = BERT_Embedding(config)
        bert_dim = self.bert_text_embedding.bert_dim

        self.qatt = QuestionAttention(bert_dim, self.att_out_dim) 

        self.hidden_layer = SimpleHidden(self.att_out_dim, self.hidden_dim)
        self.layer_output = torch.nn.Linear(self.hidden_dim, self.n_classes)


    def _init_params(self):
        self.hidden_dim = 300
        self.n_classes = 2
        self.att_out_dim = 1024

    def _read_config(self, config):
        if 'MODEL' in config:
            if 'hidden_dim' in config['MODEL']:
                self.hidden_dim = int(config['MODEL'].get('hidden_dim'))
            if 'n_classes' in config['MODEL']:
                self.n_classes = int(config['MODEL'].get('n_classes'))
            if 'att_out_dim' in config['MODEL']:
                self.sep_idx = int(config['MODEL'].get('att_out_dim'))


    def forward(self, batchItem, mask=None):
        question = batchItem[2]
        x = batchItem[0]
        #question_rep = self.bert_question_embedding(question, None)
        question_rep = self.bert_text_embedding(question, None)
        question_rep_cls = question_rep[0][:,0]

        text_rep = self.bert_text_embedding(x, None)
        text_rep = text_rep[0]

        qt_rep = self.qatt(question_rep_cls, text_rep)
        hidden = self.hidden_layer(qt_rep)
        out = self.layer_output(hidden)

        y = {
            'y_hat':out,
            'cls_att':out
        }

        return y

