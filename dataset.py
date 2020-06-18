# -*- coding: utf-8 -*-
# @Time    : 2020-06-14
# @Author  : lihang

import os
import json
import re
import codecs
import numpy as np
import torch

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


# from here by: qiwangyu
# process the text to match the word vectors
def process_text(text):
    text = text.replace(' ', '')
    text = re.sub('\[[0-9]*', '', text)
    text = re.sub('[0-9]*\]', '', text)
    text = text.replace('dis', '')
    text = text.replace('sym', '')
    text = text.replace('bod', '')
    text = text.replace('ite', '')
    text = text.replace('參', '参')
    text = text.replace('橫', '横')
    text = text.replace('？', '?')
    text = text.replace('；', ';')
    text = text.replace('：', ':')
    text = text.replace('，', ',')
    text = text.replace('—', '')
    text = text.replace('）', ')')
    text = text.replace('（', '(')

    return text


#get word vector
def get_word_vector():
    files = ['train.txt', 'dev.txt', 'test.txt']
    words = []
    for file in files:
        with open('original_dataset/' + file, 'r', encoding='utf8') as f:
            data = json.load(f)
            for line in data:
                text = line['text']
                text = process_text(text)
                for w in text:
                    words.append(w)
    pre_trained = {}
    for i, line in enumerate(codecs.open('data/wiki_100.utf8', 'r', 'utf-8')):
        line = line.rstrip().split()
        pre_trained[line[0]] = np.array([float(x) for x in line[1:]]).astype(np.float32)

    words = list(set(words))
    wv, w2i = [np.array([0 for i in range(100)])], {}
    for i in range(len(words)):
        w2i[words[i]] = i + 1
        wv.append(pre_trained[words[i]])
    wv = torch.from_numpy(np.array(wv))

    return wv, w2i


wv, w2i = get_word_vector()
MAX_LEN = 500


# generate position embedding giving locations of the disease and symptom
def generate_pos_emb(s1, e1, s2, e2):
    global MAX_LEN
    pos_emb = []
    for i in range(MAX_LEN):
        a, b = 0, 0
        if i < s1:
            a = i - s1
        elif i > e1:
            a = i - e1
        if i < s2:
            b = i - s2
        elif i > e2:
            b = i - e2
        pos_emb.append([a, b])
    return pos_emb


# convert word to vectors
def generate_train_data(text):
    global MAX_LEN
    t = [w2i[w] for w in text]
    if len(t) > MAX_LEN:
        t = t[:MAX_LEN]
    else:
        t.extend([0 for i in range(MAX_LEN - len(text))])
    return t



if __name__ == "__main__":
    bio_labeling(train_original, train_labeled)
    bio_labeling(test_original, test_labeled)
    bio_labeling(dev_original, dev_labeled)
