# -*- coding:utf-8 -*-

from pyltp import Segmentor
from gensim import corpora
import os
import numpy as np
import sys

# reload(sys)
# sys.setdefaultencoding('utf-8')

# 设置ltp文件的目录 比如: ~/ltp_data/
your_model_path = "/home/wds/ltp_data/"
MODELDIR = os.path.join(your_model_path)


def fenci(filename1, filename2):
    segmentor = Segmentor()
    segmentor.load(os.path.join(MODELDIR, "cws.model"))
    part = []
    with open(filename1, 'r') as f:
        with open(filename2, 'w') as out:
            for line in f.readlines():
                p, sentence = line.split("\t")
                part.append(p)
                words = segmentor.segment(sentence)
                out.write(" ".join(words) + "\n")
        segmentor.release()
    return part


def word_dict():
    word_dict = []
    word_list = []
    with open("/home/wds/words.txt", 'r') as wf:
        for word in wf.readlines():
            word_list.extend(word.split(" "))
    print(len(word_list))
    for item in word_list:
        if item not in word_dict:
            word_dict.append(item)
    return word_dict


def knn():
    vector_list = []  # 所有句子的向量
    allword = word_dict()  # txt中所有的词的列表
    print(len(allword))
    part = fenci("/home/wds/undo.txt", "/home/wds/done.txt")  # 每句话极性的列表
    with open("/home/wds/done.txt", 'r') as f:
        for line in f.readlines():
            word_list = []  # txt中每一行单词的列表
            line_dict = {}  # 每一行单词对应出现频率的词典
            word_list.extend(line.split(" "))
            for item in word_list:
                if item not in line_dict:
                    line_dict[item] = 1
                else:
                    line_dict[item] += 1
            vector = []  # 每句话对应的向量
            for word in allword:
                if word in line_dict.keys():
                    vector.append(line_dict[word])
                else:
                    vector.append(0)
            vector_list.append(vector)

    part_test=fenci("/home/wds/test.txt", "/home/wds/test_after.txt")

    with open("/home/wds/test_after.txt", 'r') as f:
        cnt = 1
        correct=0
        for line in f.readlines():
            test_word_list = []  # txt中每一行单词的列表
            test_line_dict = {}  # 每一行单词对应出现频率的词典
            test_word_list.extend(line.split(" "))
            for item in test_word_list:
                if item not in test_line_dict:
                    test_line_dict[item] = 1
                else:
                    test_line_dict[item] += 1
            test_vector = []  # 每句话对应的向量
            for word in allword:
                if word in test_line_dict.keys():
                    test_vector.append(test_line_dict[word])
                else:
                    test_vector.append(0)
            distance_dict = {}  # 距离和极性的词典
            i = 0
            for vector in vector_list:
                distance = np.linalg.norm(np.array(vector) - np.array(test_vector))
                distance_dict[distance] = part[i]
                i += 1
            sorted(distance_dict.items(), key=lambda e: e[0])  # 按照距离大小对词典排序
            j = 0
            middle = 0
            positive = 0
            negtive = 0
            for dis in distance_dict.keys():  # 计算距离前15个中三种情感分别的个数
                if j > 20:
                    break;
                if distance_dict[dis] is '0':
                    middle += 1;
                if distance_dict[dis] is '1':
                    positive += 1
                if distance_dict[dis] == '-1':
                    negtive += 1
                j += 1
            # print(str(positive))
            # print(str(middle))
            # print(str(negtive))
            if ((positive >= middle) and (positive >= negtive)):# 判断待测句子的极性
                pos=1
                # print("第" + str(cnt) + "个句子的极性为1")
            elif ((positive <= middle) and (negtive <= middle)):
                pos=0
                # print("第" + str(cnt) + "个句子的极性为0")
            elif ((negtive >= positive) and (negtive >= middle)):
                pos=-1
                # print("第" + str(cnt) + "个句子的极性为-1")
            if(eval(part[cnt-1])==pos):
                correct+=1
            cnt += 1
    print(len(part_test))
    print(correct)


knn()
