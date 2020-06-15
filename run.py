# -*- coding: utf-8 -*-
# @Time    : 2020-06-14
# @Author  : lihang

import os
import kashgari
from dataset import train_labeled, test_labeled, dev_labeled
from model import train_ner
from evaluate import predict_ner, evaluate_acc, export_result_and_error_sentence

bert_model_path = os.path.join('bert_model', 'chinese_L-12_H-768_A-12')
model_save_path = os.path.join('ner_model', 'ner_bert_bilstm_crf.h5')
result_save_path = os.path.join('predict', 'ner_predict.txt')
error_save_path = os.path.join('predict', 'error_sentence.txt')


def get_sequent_tagging_data(file_path):
    data_x, data_y = [], []

    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.read().splitlines()
        x, y = [], []
        for line in lines:
            rows = line.split(' ')
            if len(rows) == 1:
                data_x.append(x)
                data_y.append(y)
                x = []
                y = []
            else:
                x.append(rows[0])
                y.append(rows[1])
    return data_x, data_y


if __name__ == "__main__":

    x_train, y_train = get_sequent_tagging_data(train_labeled)
    x_valid, y_valid = get_sequent_tagging_data(dev_labeled)
    x_test, y_test = get_sequent_tagging_data(test_labeled)

    # 训练
    train_ner(x_train, y_train, x_valid, y_valid, x_test, y_test,
              sequence_length=128, epoch=3, batch_size=32,
              bert_model_path=bert_model_path, model_save_path=model_save_path)

    # 预测
    loaded_model = kashgari.utils.load_model(model_save_path)
    y_pred, predict_feature = predict_ner(loaded_model, x_test)

    # 计算准确率
    accuracy, total_accuracy = evaluate_acc(predict_feature)
    print(accuracy)
    print('total accuracy:', total_accuracy)

    # 输出结果和错误案例
    export_result_and_error_sentence(x_test, y_test, y_pred, result_save_path, error_save_path)
