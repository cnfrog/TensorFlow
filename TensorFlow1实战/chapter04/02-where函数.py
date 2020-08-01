# -*- coding: utf-8 -*-
# @Date  : 2019/3/20 22:00
# @Software : PyCharm
import tensorflow as tf

v1 = tf.constant([1.0, 2.0, 3.0, 4.0])
v2 = tf.constant([4.0, 3.0, 2.0, 1.0])

with tf.Session() as sess:
    print(tf.greater(v1, v2).eval())  # [False False  True  True]
    print(tf.where(tf.greater(v1, v2), v1, v2).eval())  # [4. 3. 3. 4.]
