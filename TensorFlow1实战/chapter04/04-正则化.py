# -*- coding: utf-8 -*-
# @Date  : 2019/3/21 21:20
# @Software : PyCharm
import tensorflow as tf

weights = tf.constant([[1.0, -2.0],[-3.0, 4.0]])
with tf.Session() as sess:
    # (|1| + |-2| + |-3| + |4| ) * 0.5 = 5
    print(sess.run(tf.contrib.layers.l1_regularizer(.5)(weights)))
    print(sess.run(tf.contrib.layers.l2_regularizer(.5)(weights)))