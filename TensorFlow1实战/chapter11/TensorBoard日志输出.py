# -*- coding: utf-8 -*-
# @Date  : 2019/4/17 9:58
# @Software : PyCharm
import tensorflow as tf

input1 = tf.constant([1.0, 2.0, 3.0], name='input1')
input2 = tf.Variable(tf.random_uniform([3]), name='input2')
output = tf.add(input1, input2, name='add')
writer = tf.summary.FileWriter("./path/log", tf.get_default_graph())
writer.close()

