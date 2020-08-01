# -*- coding: utf-8 -*-
# @Date  : 2019/3/19 22:14
# @Software : PyCharm

import tensorflow as tf

# 使用sigmoid函数将y转换为0~1之间的数据.转换后y代表预测是正样本的概率,1-y代表
# 预测是负样本的概率
y = tf.sigmoid(y)

# 定义损失函数来刻画预测值和真实值得差距
cross_entropy = -tf.reduce_mean(
    y_ * tf.log(tf.clip_by_value(y, 1e-10, 1.0))
    + (1-y)*tf.log(tf.clip_by_value(1-y, 1e-10, 1.0)))

# 定义学习率
learning_rate = 0.001
# 定义反向传播算法来优化神经网络中的参数
train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)