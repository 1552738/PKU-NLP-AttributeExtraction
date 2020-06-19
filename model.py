# -*- coding: utf-8 -*-
# @Time    : 2020-06-14
# @Author  : lihang

import kashgari
from kashgari.embeddings import BERTEmbedding
from kashgari.tasks.labeling import BiLSTM_CRF_Model
from kashgari.callbacks import EvalCallBack
import torch
import torch.nn as nn
import torch.nn.functional as F


def train_ner(x_train, y_train, x_valid, y_valid, x_test, y_test,
              sequence_length, epoch, batch_size,
              bert_model_path, model_save_path):
    """
    BERT-BiLSTM-CRF 模型训练，提取症状内部特征
    """
    bert_embedding = BERTEmbedding(bert_model_path,
                                   task=kashgari.LABELING,
                                   sequence_length=sequence_length)

    model = BiLSTM_CRF_Model(bert_embedding)

    eval_callback_val = EvalCallBack(kash_model=model,
                                     valid_x=x_valid,
                                     valid_y=y_valid,
                                     step=1)

    eval_callback_test = EvalCallBack(kash_model=model,
                                      valid_x=x_test,
                                      valid_y=y_test,
                                      step=1)

    model.fit(x_train, y_train,
              x_validate=x_valid, y_validate=y_valid,
              epochs=epoch, batch_size=batch_size,
              callbacks=[eval_callback_val, eval_callback_test])

    model.save(model_save_path)

    model.evaluate(x_test, y_test)

    return model


class CRCNN(nn.Module):
    def __init__(self, wv, pos_dim, sent_len, num_kernel, kerner_sizes, drop_prob):
        super(CRCNN, self).__init__()
        emb_dim = wv.size(1)
        self.sent_len = sent_len
        self.embed = nn.Embedding.from_pretrained(wv)
        self.pos_embed = nn.Embedding(2 * sent_len + 2, pos_dim)

        self.convs = nn.ModuleList([nn.Conv2d(1, num_kernel, (K, emb_dim + 2 * pos_dim),
                                    padding=((K - 1) // 2, 0)) for K in kerner_sizes])
        self.dropout1 = nn.Dropout(drop_prob)
        self.dropout2 = nn.Dropout(drop_prob)
        self.fc = nn.Linear(len(kerner_sizes) * num_kernel, 2)

    def forward(self, cx):
        x = cx[0]
        pos = cx[1]
        x = self.embed(x)
        x = self.dropout1(x)
        pos = torch.squeeze(pos.view(pos.shape[0], -1, 1), dim=2)
        pos = pos + self.sent_len

        p = self.pos_embed(pos)
        p = p.view(p.shape[0], p.shape[1] // 2, -1)
        x = torch.cat((x, p), 2)

        x = x.unsqueeze(1)
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs]
        x = [F.max_pool1d(line, line.size(2)).squeeze(2) for line in x]

        x = torch.cat(x, 1)
        x = self.dropout2(x)
        logit = self.fc(x)
        return logit