# -*- coding: utf-8 -*-
# @Date  : 2019/3/19 20:48
# @Software : PyCharm
import tensorflow as tf

w1 = tf.Variable(tf.random_normal([2, 3], stddev=1), name='w1')
w2 = tf.Variable(tf.random_normal([2, 2], stddev=1), name='w2')
# tf.assign(w1, w2) 不能执行
# tf.assign(w1, w2, validate_shape=False)
tf.reshape(w2,[4,1])
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run(w2))