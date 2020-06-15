# -*- coding: utf-8 -*-
# @Time    : 2020-06-14
# @Author  : lihang

import kashgari
from kashgari.embeddings import BERTEmbedding
from kashgari.tasks.labeling import BiLSTM_CRF_Model
from kashgari.callbacks import EvalCallBack


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
