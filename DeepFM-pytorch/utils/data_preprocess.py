"""
预处理数据部分

在模型的最底层是不同的特征field，因为在处理特征时，需要对离散型数据进行one-hot转化，经过one-hot
之后一列会变成多列，这样会导致特征矩阵变得非常稀疏。为应对这一问题，进行one-hot之前的每个人在都看作
一个field，这是为了在进行矩阵存储时，可以把对一个大的稀疏矩阵的存储，转换成对两个小矩阵和一个字典的存储

字典中，为离散型特征每个特征值都添加索引标识，对连续数据只给予一个特征索引，这样就不必存储离散特征的onehot
矩阵，而只需存储离散特征值对应在字典中的编号就行了，用编号作为特征标识。
"""

import sys
import math
import argparse
import hashlib, csv, math, os, pickle, subprocess

#生成特征的序号index
def gen_category_index(file_path):
    cate_dict = []
    for i in range(26):
        cate_dict.append({})
    for line in open(file_path, 'r'):
        datas = line.replace('\n','').split('\t')
        for i, item in enumerate(datas[14:]):
            if not cate_dict[i].has_key(item):
                cate_dict[i][item] = len(cate_dict[i])
    return cate_dict

#将生成的特征index写入文件
def write_category_index(file_path, cate_dict_arr):
    f = open(file_path,'w')
    for i, cate_dict in enumerate(cate_dict_arr):
        for key in cate_dict:
            f.write(str(i)+','+key+','+str(cate_dict[key])+'\n')

#加载文件中的特征index
def load_category_index(file_path):
    f = open(file_path,'r')
    cate_dict = []
    for i in range(39):
        cate_dict.append({})
    for line in f:
        datas = line.strip().split(',')
        cate_dict[int(datas[0])][datas[1]] = int(datas[2])
    return cate_dict


#从数据文件中读取特征属性
def read_raw_data(file_path, embedding_path, type):
    """
    :param file_path: string
    :param type: string (train or test)
    :return: result: dict
            result['continuous_feat']:two-dim array
            result['category_feat']:dict
            result['category_feat']['index']:two-dim array
            result['category_feat']['value']:two-dim array
            result['label']: one-dim array
    """
    begin_index = 1
    if type != 'train' and type != 'test':
        print("type error")
        return {}
    elif type == 'test':
        begin_index = 0
    cate_embedding = load_category_index(embedding_path)
    result = {'continuous_feat':[], 'category_feat':{'index':[],'value':[]}, 'label':[], 'feature_sizes':[]}
    for i, item in enumerate(cate_embedding):
        result['feature_sizes'].append(len(item))
    f = open(file_path)
    for line in f:
        datas = line.replace('\n', '').split('\t')

        indexs = []
        values = []
        flag = True
        for i, item in enumerate(datas[begin_index + 13:]):
            if not cate_embedding[i].has_key(item):
                flag = False
                break
            indexs.append(cate_embedding[i][item])
            values.append(1)
        if not flag:
            continue
        result['category_feat']['index'].append(indexs)
        result['category_feat']['value'].append(values)

        if type == 'train':
            result['label'].append(int(datas[0]))
        else:
            result['label'].append(0)

        continuous_array = []
        for item in datas[begin_index:begin_index+13]:
            if item == '':
                continuous_array.append(-10.0)
            elif float(item) < 2.0:
                continuous_array.append(float(item))
            else:
                continuous_array.append(math.log(float(item)))
        result['continuous_feat'].append(continuous_array)

    return result

#读取特征的属性字典
def read_data(file_path,emb_file):
    result = {'label':[], 'index':[],'value':[],'feature_sizes':[]}
    cate_dict = load_category_index(emb_file)
    for item in cate_dict:
        result['feature_sizes'].append(len(item))
    f = open(file_path,'r')
    for line in f:
        datas = line.strip().split(',')
        result['label'].append(int(datas[0]))
        indexs = [int(item) for item in datas[1:]]
        values = [1 for i in range(39)]
        result['index'].append(indexs)
        result['value'].append(values)
    return result

#获取特征的embedding
def gen_category_emb_from_libffmfile(filepath, dir_path):
    fr = open(filepath)
    cate_emb_arr = [{} for i in range(39)]
    for line in fr:
        datas = line.strip().split(' ')
        for item in datas[1:]:
            [filed, index, value] = item.split(':')
            filed = int(filed)
            index = int(index)
            if not cate_emb_arr[filed].has_key(index):
                cate_emb_arr[filed][index] = len(cate_emb_arr[filed])

    with open(dir_path, 'w') as f:
        for i,item in enumerate(cate_emb_arr):
            for key in item:
                f.write(str(i)+','+str(key)+','+str(item[key])+'\n')

#生成特征的embedding文件
def gen_emb_input_file(filepath, emb_file, dir_path):
    cate_dict = load_category_index(emb_file)
    fr = open(filepath,'r')
    fw = open(dir_path,'w')
    for line in fr:
        row = []
        datas = line.strip().split(' ')
        row.append(datas[0])
        for item in datas[1:]:
            [filed, index, value] = item.split(':')
            filed = int(filed)
            row.append(str(cate_dict[filed][index]))
        fw.write(','.join(row)+'\n')


#      
# cate_dict = gen_category_index('../data/train.txt')
#将特征字典写入category_index文件
# write_category_index('../data/category_index.csv',cate_dict)

#从训练集中获取cate_dict
# result_dict = read_raw_data('../data/train.txt', '../data/category_index.csv', 'train')
#  
#train.ffm和test.ffm文件由https://github.com/guestwalk/kaggle-2014-criteo链接中的脚本生成
#gen_category_emb_from_libffmfile('../data/train.ffm','../data/category_emb1.csv')
#gen_category_emb_from_libffmfile('../data/test.ffm','../data/category_emb2.csv')

#生成类型特征，写入文件
#gen_emb_input_file('../data/train.ffm','../data/category_emb1.csv','../data/train_input.csv')
#gen_emb_input_file('../data/test.ffm','../data/category_emb2.csv','../data/test_input.csv')
#可将两个category_emb进行拼接，成为一个文件category_emb.csv


#读入类型特征embedding，返回数据的特征属性字典
# train_result = read_data('../data/train_input.csv', '../data/category_emb.csv')
# test_result = read_data('../data/test_input.csv', '../data/category_emb.csv')


