# -*- coding: utf-8 -*-
# @Date  : 2019/3/19 10:58
# @Software : PyCharm

import tensorflow as tf


a = tf.constant([1.0, 2.0], name='a')
b = tf.constant([2.0, 3.0], name='b')
ret = a + b
print(ret.graph is tf.get_default_graph())
