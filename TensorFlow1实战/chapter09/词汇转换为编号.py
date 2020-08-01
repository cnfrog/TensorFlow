# -*- coding: utf-8 -*-
# @Date  : 2019/4/9 21:33
# @Software : PyCharm
import tensorflow as tf
import codecs
import sys

RAW_DATA = './data/ptb.train.txt' # 训练集数据文件
VOCAB_OUTPUT = 'ptb.vocab' # 输出的词汇表文件
OUTPUT_DATA = 'ptb.train' # 将单词替换为单词编号后的输出文件

# 读取词汇表,并建立词汇到单词编号的映射