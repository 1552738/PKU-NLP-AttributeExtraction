# -*- coding: utf-8 -*-
# @Time    : 2020-06-14
# @Author  : lihang

import os
import kashgari
from dataset import *
from model import train_ner, CRCNN
from evaluate import predict_ner, evaluate_acc, export_result_and_error_sentence
import torch
import sys
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import codecs
import numpy as np
import json

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


# by: qiwangyu
# get dataset
class my_dataset(Dataset):
    def __init__(self, train_data, train_pos, labels):
        self.train_data = torch.from_numpy(train_data).long()
        self.train_pos = torch.from_numpy(train_pos).long()
        self.labels = torch.from_numpy(labels).long()

    def __getitem__(self, idx):
        return (self.train_data[idx], self.train_pos[idx], self.labels[idx])

    def __len__(self):
        return len(self.labels)


if __name__ == "__main__":

    if sys.argv[1] == 'dis':
        # by: qiwangyu
        # get train data for disease
        train_data, train_pos, labels = [], [], []
        with open('original_dataset/train.txt', 'r', encoding='utf8') as f:
            data = json.load(f)
            for line in data:
                dis = re.findall('\[[^\[\]]*\]dis', line['text'])
                text = process_text(line['text'])

                # find all diseases in text, like {'NICH': [0,3]}
                pos_dis = {}
                for d in dis:
                    ss = re.findall('\[[0-9]+', d)
                    ee = re.findall('[0-9]+\]', d)
                    if not (ss and ee):
                        continue
                    s = int(ss[0][1:])
                    e = int(ee[0][:-1])
                    if s < 0 or s > e or e > MAX_LEN:
                        continue
                    pos_dis[text[s: e + 1]] = [s, e]

                # generate train data for each dis and symptom
                for i, sym in line['symptom'].items():
                    if sym['has_problem']:
                        continue
                    if len(sym['self']['pos']) < 2:
                        continue
                    s, e = sym['self']['pos']
                    if s < 0 or s > e or e > MAX_LEN:
                        continue
                    sym_dis = []
                    if len(sym['disease']['pos']) > 2:
                        sym_dis = sym['disease']['val'].split()
                    else:
                        sym_dis.append(sym['disease']['val'])
                    for d, p in pos_dis.items():
                        train_data.append(generate_train_data(text))
                        train_pos.append(generate_pos_emb(s, e, p[0], p[1]))
                        if d in sym_dis:
                            labels.append(1)
                            for k in range(len(pos_dis) - 1):
                                train_data.append(generate_train_data(text))
                                train_pos.append(generate_pos_emb(s, e, p[0], p[1]))
                                labels.append(1)
                        else:
                            labels.append(0)

        train_data = np.array(train_data)
        train_pos = np.array(train_pos)
        labels = np.array(labels)

        train_dataset = my_dataset(train_data, train_pos, labels)
        train_loader = DataLoader(train_dataset, batch_size=48, shuffle=True)

        # set parameters
        num_epochs = 10
        model = CRCNN(wv, pos_dim=100, sent_len=MAX_LEN, num_kernel=100, kerner_sizes=[2, 3, 4, 5], drop_prob=0.5)
        model = model.double()
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        lr = 0.01
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)
        loss = nn.CrossEntropyLoss()
        step, epoch = 0, 0
        MODEL_PATH = 'disease_model/crcnn_model.ckpt'

        while epoch != num_epochs:
            epoch += 1
            total, right = 0, 0
            loss_all, cur_step = 0, 0
            for x0, x1, y in train_loader:
                step += 1
                optimizer.zero_grad()
                x0, x1, y = x0.to(device), x1.to(device), y.to(device)
                y_ = model([x0, x1])
                l = loss(y_, y)
                loss_all += l
                cur_step += 1
                right += (y_.argmax(dim=1) == y).float().sum().item()
                total += len(y)

                l.backward()
                optimizer.step()
                if step % 100 == 0:
                    print("epoch:%d\t step:%d\t loss:%f" % (epoch, step, loss_all / cur_step))
            print("epoch:%d\t acc:%f" % (epoch, right / total))
        torch.save(model, MODEL_PATH)

    else:
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
