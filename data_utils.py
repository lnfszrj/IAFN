# -*- coding: utf-8 -*-
# file: data_utils.py
# author: jyu5 <yujianfei1990@gmail.com>
# Copyright (C) 2018. All Rights Reserved.

import os
import pickle
import numpy as np
import torch.nn.init
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import metrics

N = 5  # 维度


def distance_fun(p1, p2, N):
    result = 0
    for i in range(0, N):
        result = result + ((p1[i] - p2[i]) ** 2)
    return np.sqrt(result)


def mean_fun(a):
    return np.mean(a, axis=0)


def farthest(center_arr, arr):
    f = [0, 0]
    max_d = 0
    for e in arr:
        d = 0
        for i in range(center_arr.__len__()):
            d = d + np.sqrt(distance_fun(center_arr[i], e, N))
        if d > max_d:
            max_d = d
            f = e
    return f


def closest(a, arr):
    c = arr[1]
    min_d = distance_fun(a, arr[1])
    arr = arr[1:]
    for e in arr:
        d = distance_fun(a, e)
        if d < min_d:
            min_d = d
            c = e
    return c


def load_word_vec(path, word2idx=None):
    fin = open(path, 'r', encoding='utf-8', newline='\n', errors='ignore')
    word_vec = {}
    for line in fin:
        tokens = line.rstrip().split(' ')
        if word2idx is None or tokens[0] in word2idx.keys():
            word_vec[tokens[0]] = np.asarray(tokens[1:], dtype='float32')
    return word_vec


def build_embedding_matrix(word2idx, embed_dim, type):
    embedding_matrix_file_name = '{0}_{1}_embedding_matrix.dat'.format(str(embed_dim), type)
    load_embedding_boolean = True
    if load_embedding_boolean:
        print('loading word vectors...')
        embedding_matrix = np.zeros((len(word2idx) + 2, embed_dim))  # idx 0 and len(word2idx)+1 are all-zeros
        # fname = '/home/jfyu/torch/stanford_treelstm-master/data/glove/glove.twitter.27B.' + str(embed_dim) + 'd.txt' \
        # if embed_dim != 300 else '/home/jfyu/torch/stanford_treelstm-master/data/glove/glove.840B.300d.txt'
        # fname = '../../../pytorch/glove.twitter.27B.' + str(embed_dim) + 'd.txt'
        fname = '/home/xinke901/zrj/zrj/TASL2021/TASL2020/ESAFN-master/glove.840B.300d.txt'
        # if embed_dim != 200 else '../../../pytorch/glove.6B.300d.txt'
        word_vec = load_word_vec(fname, word2idx=word2idx)
        print('building embedding_matrix:', embedding_matrix_file_name)
        for word, i in word2idx.items():
            vec = word_vec.get(word)
            if vec is not None:
                # words not found in embedding index will be all-zeros.
                # print(embedding_matrix.shape)
                # print(vec.shape)
                embedding_matrix[i] = vec
        pickle.dump(embedding_matrix, open(embedding_matrix_file_name, 'wb'))
    return embedding_matrix


def image_process(image_path, transform):
    image = Image.open(image_path).convert('RGB')
    # print("函数内")
    # print(image_path)
    # print(image.size)
    image = transform(image)
    return image


class Tokenizer(object):
    def __init__(self, lower=False, max_seq_len=None, max_aspect_len=None):
        self.lower = lower
        self.max_seq_len = max_seq_len
        self.max_aspect_len = max_aspect_len
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 2
        self.word2idx['ttttt'] = 1
        self.idx2word[1] = 'ttttt'

    def fit_on_text(self, text):
        if self.lower:
            text = text.lower()
        words = text.split()
        for word in words:
            if word not in self.word2idx:
                self.word2idx[word] = self.idx
                self.idx2word[self.idx] = word
                self.idx += 1

    @staticmethod
    def pad_sequence(sequence, maxlen, dtype='int64', padding='pre', truncating='pre', value=0.):
        x = (np.ones(maxlen) * value).astype(dtype)
        if truncating == 'pre':
            trunc = sequence[-maxlen:]
        else:
            trunc = sequence[:maxlen]
        trunc = np.asarray(trunc, dtype=dtype)
        if padding == 'post':
            x[:len(trunc)] = trunc
        else:
            x[-len(trunc):] = trunc
        return x

    def text_to_sequence(self, text, reverse=False):
        if self.lower:
            text = text.lower()
        words = text.split()
        unknownidx = len(self.word2idx) + 1
        sequence = [self.word2idx[w] if w in self.word2idx else unknownidx for w in words]
        if len(sequence) == 0:
            sequence = [0]
        pad_and_trunc = 'post'  # use post padding together with torch.nn.utils.rnn.pack_padded_sequence
        if reverse:
            sequence = sequence[::-1]
        return Tokenizer.pad_sequence(sequence, self.max_seq_len, dtype='int64', padding=pad_and_trunc,
                                      truncating=pad_and_trunc)


class ABSADataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


class ABSADatesetReader:
    @staticmethod
    def __read_text__(fnames):
        text = ''
        for fname in fnames:
            fin = open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
            lines = fin.readlines()
            fin.close()
            for i in range(0, len(lines), 4):
                text_left, _, text_right = [s.lower().strip() for s in lines[i].partition("$T$")]
                aspect = lines[i + 1].lower().strip()
                text_raw = text_left + " " + aspect + " " + text_right
                text += text_raw + " "
        return text

    @staticmethod
    def __read_data__(fname, tokenizer, path_img, transform):
        print('--------------' + fname + '---------------')
        fin = open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
        lines = fin.readlines()
        fin.close()

        all_data = []
        count = 0

        IMAGE = []
        for i in range(0, len(lines), 4):
            image_id = lines[i + 3].strip()
            image_name = image_id
            image_path = os.path.join(path_img, image_name)

            if not os.path.exists(image_path):
                print(image_path)
            try:
                image = image_process(image_path, transform)
            except:
                image_path_fail = os.path.join(path_img, '17_06_4705.jpg')
                image = image_process(image_path_fail, transform)

            IMAGE.append(image)

        # print("len(IMAGE):", len(IMAGE))
        # print("IMAGE[0].size():", IMAGE[0].size())

        IMAGE_numpy = []
        for item in IMAGE:
            n_item = item.numpy()
            IMAGE_numpy.append(n_item)

        temp = np.array(IMAGE_numpy)
        arr_ = temp.reshape(len(IMAGE), 3 * 224 * 224)
        # print(temp.shape)
        # print("切片前：", arr_.shape)

        arr = arr_[:, 0:150000:30000]
        # print("切片后：", arr.shape)

        # 开始KMeans聚类！！！

        K = 6
        r = np.random.randint(arr.__len__() - 1)
        center_arr = np.array([arr[r]])
        cla_arr = [[]]
        cla_arr_ = [[]]
        for i in range(K - 1):
            k = farthest(center_arr, arr)
            center_arr = np.concatenate([center_arr, np.array([k])])
            cla_arr.append([])

        for i in range(K - 1):
            cla_arr_.append([])

        ## 迭代聚类
        numb = 0
        cla_temp_ = cla_arr_
        # print(len(cla_temp_))

        n = 20
        cla_temp = cla_arr

        IMAGE_CLASS = []

        for i in range(n):
            for e in arr:
                ki = 0
                min_d = distance_fun(e, center_arr[ki], N)
                for j in range(1, center_arr.__len__()):
                    if distance_fun(e, center_arr[j], N) < min_d:
                        min_d = distance_fun(e, center_arr[j], N)
                        ki = j
                cla_temp[ki].append(e)
                # print("cla_temp[ki]:", ki, "——", len(cla_temp[ki]))

                IMAGE_CLASS.append(ki)

                # 自己添加的，分类后的原始图像！
                # print("类别:", ki)
                # print("长度：", len(IMAGE))
                # print("索引：", numb)

                it = IMAGE[numb]
                cla_temp_[ki].append(it)
                numb += 1

            numb = 0

            for k in range(center_arr.__len__()):
                if n - 1 == i:
                    break
                center_arr[k] = mean_fun(cla_temp[k])
                cla_temp[k] = []
                cla_temp_[k] = []
                IMAGE_CLASS = []

        sil_score = metrics.silhouette_score(arr, IMAGE_CLASS, metric='cosine')       # 计算轮廓系数
        print("轮廓系数：", sil_score)
        dbi_score = metrics.davies_bouldin_score(arr, IMAGE_CLASS)                # 计算DBI系数
        print("DBI系数：", dbi_score)

        # print(len(cla_temp_))       # 此时的列表 cla_temp_ ，长度为分类个数（3），其中储存的是每一类的原始torch数据！！
        # print(cla_temp_[0][0].shape)    # 分好类的一个图片样本，第0类的第1个（索引为0）数据样本  torch.Size([3, 224, 224])
        # print(len(cla_temp_[0]))
        # print(len(cla_temp[0]))     # 仅供调试使用
        # print("分类数组长度：", len(IMAGE_CLASS))

        # print(center_arr)
        # print(center_arr.shape)
        #
        # print(cla_temp)
        # print(len(cla_temp))

        # 以下为KMeans作图可视化的代码：

        if N >= 2:
            print(N, '维数据前两维投影')
        col = ['gold', 'blue', 'violet', 'cyan', 'red', 'lime', 'brown', 'black', 'silver']
        plt.figure(figsize=(10, 10))
        for i in range(K):
            plt.scatter(center_arr[i][0], center_arr[i][1], color=col[i])
            plt.scatter([e[0] for e in cla_temp[i]], [e[1] for e in cla_temp[i]], color=col[i])
        plt.show()

        if N >= 3:
            print(N, '维数据前三维投影')
            fig = plt.figure(figsize=(8, 8))
            ax = Axes3D(fig)
            for i in range(K):
                ax.scatter(center_arr[i][0], center_arr[i][1], center_arr[i][2], color=col[i])
                ax.scatter([e[0] for e in cla_temp[i]], [e[1] for e in cla_temp[i]], [e[2] for e in cla_temp[i]],
                           color=col[i])
            plt.show()

        # print(N, '维')
        # for i in range(K):
        #     print('第', i + 1, '个聚类中心坐标：')
        #     for j in range(0, N):
        #         print(center_arr[i][j])

        '''
                IMAGE = []

                 for i in range(0, len(lines), 4):
                    image_id = lines[i + 3].strip()
                    image_name = image_id
                    image_path = os.path.join(path_img, image_name)

                    if not os.path.exists(image_path):
                        print(image_path)
                    try:
                        image = image_process(image_path, transform)
                    except:
                        image_path_fail = os.path.join(path_img, '17_06_4705.jpg')
                        image = image_process(image_path_fail, transform)

                    IMAGE.append(image)

                    # 对IMAGE列表中的image向量进行分类/聚类...
                    # ...
                    # ...
                    # trait_class = [....]

                '''

        for i in range(0, len(lines), 4):
            text_left, _, text_right = [s.lower().strip() for s in lines[i].partition("$T$")]
            aspect = lines[i + 1].lower().strip()
            polarity = lines[i + 2].strip()
            image_id = lines[i + 3].strip()

            if text_left == "":
                text_left_for_tdan = "$unk$"
            else:
                text_left_for_tdan = text_left
            if text_right == "":
                text_right_for_tdan = "$unk$"
            else:
                text_right_for_tdan = text_right

            text_left_for_fusion = text_left + " ttttt"
            text_right_for_fusion = "ttttt " + text_right

            text_raw_indices = tokenizer.text_to_sequence(text_left + " " + aspect + " " + text_right)
            if text_left == "" and text_right == "":
                text_raw_without_aspect_indices = tokenizer.text_to_sequence("$unk$")
            else:
                text_raw_without_aspect_indices = tokenizer.text_to_sequence(text_left + " " + text_right)
            text_left_indices = tokenizer.text_to_sequence(text_left_for_tdan)
            text_left_indicator = tokenizer.text_to_sequence(text_left_for_fusion)
            text_left_with_aspect_indices = tokenizer.text_to_sequence(text_left + " " + aspect)
            text_right_indices = tokenizer.text_to_sequence(text_right_for_tdan, reverse=True)
            text_right_indicator = tokenizer.text_to_sequence(text_right_for_fusion)
            text_right_with_aspect_indices = tokenizer.text_to_sequence(" " + aspect + " " + text_right, reverse=True)
            aspect_indices = tokenizer.text_to_sequence(aspect)
            polarity = int(polarity) + 1

            image_name = image_id
            image_path = os.path.join(path_img, image_name)
            # print(image_path)

            # if not os.path.exists(image_path):
            #     print(image_path)
            # try:
            #     # image = image_process("/home/xinke901/zrj/zrj/TASL2021/TASL2020/ESAFN-master/IJCAIdata/twitter2015_images/73066.jpg", transform)
            #     image = image_process(image_path, transform)
            # # print(image.size())
            # except:
            #     count += 1
            #     # print('image has problem!')
            #     image_path_fail = os.path.join(path_img, '17_06_4705.jpg')
            #     image = image_process(image_path_fail, transform)

            # print("image.size():", image.size())

            sub = 0
            index1 = 0
            index2 = 0
            index3 = 0

            image1 = torch.zeros([3, 224, 224])
            image2 = torch.zeros([3, 224, 224])
            image3 = torch.zeros([3, 224, 224])

            if IMAGE_CLASS[sub] == 0:
                image1 = cla_temp_[0][index1]
                index1 += 1
            if IMAGE_CLASS[sub] == 1:
                image2 = cla_temp_[1][index2]
                index2 += 1
            if IMAGE_CLASS[sub] == 2:
                image3 = cla_temp_[2][index3]
                index3 += 1

            sub += 1

            data = {
                'text_raw_indices': text_raw_indices,
                'text_raw_without_aspect_indices': text_raw_without_aspect_indices,
                'text_left_indices': text_left_indices,
                'text_left_indicator': text_left_indicator,
                'text_left_with_aspect_indices': text_left_with_aspect_indices,
                'text_right_indices': text_right_indices,
                'text_right_indicator': text_right_indicator,
                'text_right_with_aspect_indices': text_right_with_aspect_indices,
                'aspect_indices': aspect_indices,
                'polarity': polarity,
                'image1': image1,
                'image2': image2,
                'image3': image3,
            }  # 非对应的分类图像为Tensor(0)

            all_data.append(data)

        other = len(all_data) % 10
        while other > 0:
            del all_data[-1]
            other -= 1

        print("all_data的长度：", len(all_data))
        print('the number of problematic samples: ' + str(count))
        return all_data

    def __init__(self, transform, dataset='twitter2017', embed_dim=300, max_seq_len=40, path_image='./twitter_subimages'):
        print("preparing {0} dataset...".format(dataset))
        fname = {
            'twitter': {
                'train': './datasets/twitter/train.txt',
                'dev': './datasets/twitter/dev.txt',
                'test': './datasets/twitter/test.txt'
            },
            'twitter2015': {
                'train': './datasets/twitter2015/train.txt',
                'dev': './datasets/twitter2015/dev.txt',
                'test': './datasets/twitter2015/test.txt'
            },
            'twitter2017': {
                'train': '/home/xinke901/zrj/zrj/TASL2021/TASL2020/ESAFN-master/IJCAIdata/twitter2017/train.txt',
                'dev': '/home/xinke901/zrj/zrj/TASL2021/TASL2020/ESAFN-master/IJCAIdata/twitter2017/dev.txt',
                'test': '/home/xinke901/zrj/zrj/TASL2021/TASL2020/ESAFN-master/IJCAIdata/twitter2017/test.txt'
            },
            'snap': {
                'train': './datasets/snap/train.txt',
                'dev': './datasets/snap/dev.txt',
                'test': './datasets/snap/test.txt'
            }
        }
        text = ABSADatesetReader.__read_text__([fname[dataset]['train'], fname[dataset]['dev'], fname[dataset]['test']])
        tokenizer = Tokenizer(max_seq_len=max_seq_len)
        tokenizer.fit_on_text(text.lower())
        self.embedding_matrix = build_embedding_matrix(tokenizer.word2idx, 300, dataset)
        self.train_data = ABSADataset(
            ABSADatesetReader.__read_data__(fname[dataset]['train'], tokenizer, path_image, transform))
        self.dev_data = ABSADataset(
            ABSADatesetReader.__read_data__(fname[dataset]['dev'], tokenizer, path_image, transform))
        self.test_data = ABSADataset(
            ABSADatesetReader.__read_data__(fname[dataset]['test'], tokenizer, path_image, transform))
