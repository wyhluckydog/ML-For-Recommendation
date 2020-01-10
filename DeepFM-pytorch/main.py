# -*- coding:utf-8 -*-
'''
DeepFM的主程序
'''

from utils import data_preprocess
from DeepFM import DeepFM
import torch

#读入训练集的特征字典数据
result_dict = data_preprocess.read_data('./data/train_input.csv', './data/category_emb.csv')
#读入测试集的特征字典数据
test_dict = data_preprocess.read_data('./data/test_input.csv', './data/category_emb.csv')


deepfm = DeepFM(39,result_dict['feature_sizes'],verbose=True, weight_decay=0.0001,use_fm=True,use_deep=True)
deepfm.fit(result_dict['index'], result_dict['value'], result_dict['label'],
               test_dict['index'], test_dict['value'], test_dict['label'],ealry_stopping=True,refit=True)
