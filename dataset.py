# -*- coding: utf-8 -*-
# @Time    : 2020-06-14
# @Author  : lihang

import os
import json

feature = ['disease', 'body', 'subject', 'decorate', 'frequency', 'item']
label = ['DIS', 'BOD', 'SUB', 'DEC', 'FRE', 'ITE']

train_original = os.path.join('original_dataset', 'train.txt')
test_original = os.path.join('original_dataset', 'test.txt')
dev_original = os.path.join('original_dataset', 'dev.txt')

train_labeled = os.path.join('data', 'symptom.train')
test_labeled = os.path.join('data', 'symptom.test')
dev_labeled = os.path.join('data', 'symptom.dev')


def bio_labeling(original_file, labeled_file):
    """
    对原始数据中的症状进行 BIO 序列标注
    """
    f = open(original_file, 'r')
    json_format = json.loads(f.read())
    f.close()

    for line in json_format:
        symptoms = line['symptom']
        for symptom, detail in symptoms.items():
            if detail['has_problem']:
                continue

            # 对症状名称进行预处理
            self_val = detail['self']['val']
            if self_val == '':
                self_val = symptom.split(' ')[1]
            self_val = self_val.replace(' ', '')
            self_val = self_val.replace('[32', '')
            self_val = self_val.replace('36]', '')
            self_val = self_val.replace('[', '')
            self_val = self_val.replace(']', '')
            self_val = self_val.replace('bod', '')

            sequence = ['O'] * len(self_val)

            # 标注除疾病外的几个特征
            for i in range(1, len(feature)):
                if detail[feature[i]]['val'] == '':
                    continue
                val_list = detail[feature[i]]['val'].split(' ')
                for v in val_list:
                    beg = self_val.find(v)
                    if beg != -1:
                        sequence[beg] = 'B-' + label[i]
                        for j in range(1, len(v)):
                            sequence[beg + j] = 'I-' + label[i]

            f = open(labeled_file, 'a')
            for i in range(len(sequence)):
                f.write(self_val[i] + ' ' + sequence[i] + '\n')
            f.write('\n')
            f.close()


if __name__ == "__main__":
    bio_labeling(train_original, train_labeled)
    bio_labeling(test_original, test_labeled)
    bio_labeling(dev_original, dev_labeled)
