# -*- coding: utf-8 -*-
# @Date  : 2019/4/17 9:36
# @Software : PyCharm
import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

tf.logging.set_verbosity(tf.logging.INFO)


# 通过 tf.layers 来定义模型结构。这里可以使用原生态 TensorFlow API 或者任何
# TensorFlow 的高层封装。 x 给出了输入层张量， is_training 指明了是否为训练。该函数返回前向传播的结果
def lenet(x, is_training):
    # 将输入转换为卷积层需要的形状
    x = tf.reshape(x, [-1, 28, 28, 1])
    net = tf.layers.conv2d(x, 32, 5, activation=tf.nn.relu)
    net = tf.layers.MaxPooling2D(net, 2, 2)
    net = tf.layers.conv2d(net, 64, 3, activation=tf.nn.relu)
    net = tf.layers.MaxPooling2D(net, 2, 2)
    net = tf.contrib.layers.flatten(net)
    net = tf.layers.dropout(net, rate=0.4, training=is_training)
    return tf.layers.dense(net, 10)


