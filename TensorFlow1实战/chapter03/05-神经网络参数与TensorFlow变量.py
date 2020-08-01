# -*- coding: utf-8 -*-
# @Date  : 2019/3/19 17:56
# @Software : PyCharm

import tensorflow as tf
# 一个2X3的矩阵,均值为0,标准差为2的随机数
weight = tf.Variable(tf.random_normal([1, 2], stddev=2, mean=1, dtype=tf.float32))
biases = tf.Variable(tf.zeros([3])) # [0 0 0]
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run(weight.value()))
    print(sess.run(biases.value()))