# -*- coding: utf-8 -*-
# @Time    : 2020-06-14
# @Author  : lihang

import json
import numpy as np
from dataset import test_original

feature = ['BOD', 'DEC', 'FRE', 'ITE', 'SUB']


def extract_test_feature():
    """
    原始数据症状内部特征解析
    """
    f = open(test_original, 'r')
    test_json = json.loads(f.read())
    f.close()

    test_feature = [[], [], [], [], []]
    for line in test_json:
        symptoms = line['symptom']
        for symptom, detail in symptoms.items():
            if detail['has_problem']:
                continue

            test_feature[0].append(detail['body']['val'])
            test_feature[1].append(detail['decorate']['val'])
            test_feature[2].append(detail['frequency']['val'])
            test_feature[3].append(detail['item']['val'])
            test_feature[4].append(detail['subject']['val'])

    return test_feature


def extract_labels(text, ners):
    """
    NER识别结果解析
    """
    ner_reg_list = []
    if ners:
        new_ners = []
        for ner in ners:
            new_ners += ner
        for word, tag in zip([char for char in text], new_ners):
            if tag != 'O':
                ner_reg_list.append((word, tag))

    # 输出模型的NER识别结果
    labels = {}
    if ner_reg_list:
        for i, item in enumerate(ner_reg_list):
            if item[1].startswith('B'):
                label = ""
                end = i + 1
                while end <= len(ner_reg_list) - 1 and ner_reg_list[end][1].startswith('I'):
                    end += 1

                ner_type = item[1].split('-')[1]

                if ner_type not in labels.keys():
                    labels[ner_type] = []

                label += ''.join([item[0] for item in ner_reg_list[i:end]])
                labels[ner_type].append(label)

    return labels


def predict_ner(model, x_test):
    """
    症状内部特征预测
    """
    y_pred = model.predict(x_test)

    # 将预测序列进行整理
    predict_feature = [[], [], [], [], []]
    for index in range(len(x_test)):
        line = ''.join(x_test[index])
        labels = extract_labels(line, [y_pred[index]])
        for i in range(len(feature)):
            fea = labels.get(feature[i])
            if fea is None:
                predict_feature[i].append('')
            else:
                predict_feature[i].append(' '.join(fea))

    return y_pred, predict_feature


def evaluate_acc(predict_feature):
    """
    计算测试集上每个特征的准确率以及全匹配准确率（不包含disease）
    """
    test_feature = extract_test_feature()
    # 每个特征上的准确率
    accuracy = {}
    for i in range(len(test_feature)):
        one_feature_correct_count = 0
        for j in range(len(test_feature[i])):
            if test_feature[i][j] == predict_feature[i][j]:
                one_feature_correct_count += 1
        accuracy[feature[i]] = one_feature_correct_count / len(test_feature[i])

    # 全匹配准确率（不包含disease）
    test_feature = np.array(test_feature).T
    predict_feature = np.array(predict_feature).T
    correct_count = 0
    total_count = len(test_feature)
    for i in range(total_count):
        if (test_feature[i] == predict_feature[i]).all():
            correct_count += 1
    total_accuracy = correct_count / total_count

    return accuracy, total_accuracy


def export_result_and_error_sentence(x_test, y_test, y_pred, result_save_path, error_save_path):
    """
    输出预测结果文件以及错误案例
    :return:
    """
    f = open(result_save_path, 'w')
    f_err = open(error_save_path, 'w')

    for i in range(len(x_test)):
        err_flag = False
        for j in range(len(x_test[i])):
            f.write(x_test[i][j] + ' ' + y_test[i][j] + ' ' + y_pred[i][j] + '\n')
            if y_test[i][j] != y_pred[i][j]:
                err_flag = True
        f.write('\n')
        if err_flag:
            for j in range(len(x_test[i])):
                f_err.write(x_test[i][j] + ' ' + y_test[i][j] + ' ' + y_pred[i][j] + '\n')
            f_err.write('\n')

    f.close()
    f_err.close()