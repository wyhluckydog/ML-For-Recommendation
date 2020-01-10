# -*- coding:utf-8 -*-

"""
本程序用于按比例划分dateset为训练集和测试集
在命令行中输入：python split.py s_path tr_path te_path prob

"""

import argparse
import sys
import random

s_path = sys.argv[1]      #dataset文件路径
tr_path = sys.argv[2]     #生成的训练集的保存路径
te_path = sys.argv[3]     #生成的测试集的保存路径
prob = float(sys.argv[4])       #设置训练集和测试集的划分比例

with open(tr_path,'wb') as fr:
    with open(te_path,'wb') as fe:
        for line in open(s_path,'rb'):
            if random.random() < prob:
                fr.write(line)
            else:
                fe.write(line)
