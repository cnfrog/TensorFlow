# -*- coding: utf-8 -*-
# @Date  : 2019/3/19 18:30
# @Software : PyCharm

import tensorflow as tf

# seed参数设定了随机种子,这样可以保证每次运行得到的结果是一样的
w1 = tf.Variable(tf.random_normal((2, 3), stddev=1, seed=1))
w2 = tf.Variable(tf.random_normal((3, 1), stddev=1, seed=1))

# 暂时将输入的特征向量定义为一个常量,注意这里的x是一个1X2的矩阵
x = tf.constant([[0.7, 0.9]])

a = tf.matmul(x ,w1)
y = tf.matmul(a, w2)

# sess = tf.Session()
# sess.run(w1.initializer)
# sess.run(w2.initializer)
#
# print(sess.run(y))
# sess.close()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run(y))
    print(tf.global_variables()) # 获取当前计算图上所有的变量
    print(tf.trainable_variables()) # 获取当前计算图上所有需要优化的参数