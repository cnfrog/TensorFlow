# -*- coding: utf-8 -*-
# @Date  : 2019/4/15 21:26
# @Software : PyCharm
import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

# 将TensorFlow日志信息输出到屏幕
tf.logging.set_verbosity(tf.logging.INFO)
mnist = input_data.read_data_sets('./path/MNIST_data', one_hot=True)

# 指定神经网络的输入层,所有这里指定的输入都会拼接在一起作为整个神经网络的输入
feature_columns = [tf.feature_column.numeric_column('image', shape=[784])]

estimator = tf.estimator.DNNClassifier(feature_columns=feature_columns,
                                       hidden_units=[500],
                                       n_classes=10,
                                       optimizer=tf.train.AdamOptimizer(),
                                       model_dir='./path/log')

train_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={'image': mnist.train.images},
    y=mnist.train.labels.astype(np.int32),
    num_epochs=None,
    batch_size=128,
    shuffle=True
)
estimator.train(input_fn=train_input_fn, steps=10000)

# 测试
# test_input_fn = tf.estimator.inputs.numpy_input_fn(
#     x={'image': mnist.test.images},
#     y=mnist.test.labels.astype(np.int32),
#     num_epochs=1,
#     batch_size=128,
#     shuffle=False
# )
# accuracy_score = estimator.evaluate(input_fn=test_input_fn)['accuracy']
# print('\nTest accuracy:%g %%' % (accuracy_score * 100))
