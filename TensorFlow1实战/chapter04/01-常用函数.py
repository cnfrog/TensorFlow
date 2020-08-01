# -*- coding: utf-8 -*-
# @Date  : 2019/3/20 19:54
# @Software : PyCharm

import tensorflow as tf

v = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
l = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
with tf.Session() as sess:
    print(tf.clip_by_value(v, 2.5, 4.5).eval())  # [[2.5 2.5 3. ][4.  4.5 4.5]]
    print(tf.log(l).eval())  # [0.        0.6931472 1.0986123]
    print(tf.reduce_mean(l).eval())

# softmax回归之后的交叉熵损失函数
# cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=y_)
