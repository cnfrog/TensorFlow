# -*- coding: utf-8 -*-
# @Date  : 2019/4/14 11:01
# @Software : PyCharm
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np

from tensorflow.examples.tutorials.mnist import input_data


# 通过Tensorflow-Slim来定义LeNet-5的网络结构
def lenet5(inputs):
    # 将输入数据转换为一个四维数组,其中第一维表示batch大小,另三维表示一张图片
    inputs = tf.reshape(inputs, [-1, 28, 28, 1])
    # 定义第一层卷积层,该卷积层的深度为32,过滤器的大小为5x5,使用全0填充
    net = slim.conv2d(inputs, 32, [5, 5], padding='SAME', scope='layer1-conv')

    # 定义一个最大池化层,其过滤器大小为2x2,步长为2
    net = slim.max_pool2d(net, 2, stride=2, scope='layer2-max-pool')

    # 类似的定义其他网络层结构
    net = slim.conv2d(net, 64, [5, 5], padding='SAME', scope='layer3-conv')
    net = slim.max_pool2d(net, 2, stride=2, scope='layer4-max-pool')

    # 直接使用TensorFlow-slim封装好的flatten函数将4维矩阵转为2维,这样可以方便
    # 后面的全连接层的计算,通过封装好的函数,用户不再需要自己计算通过卷积层之后矩阵的大小
    net = slim.flatten(net, scope='flatten')
    # 通过TensorFlow-Slim定义全连接层,该连接层有500个隐藏节点
    net = slim.fully_connected(net, 500, scope='layer5')
    net = slim.fully_connected(net, 10, scope='output')
    return net


# 通过TensorFlow-slim定义网络结构,并使用之前章节中给出的方式训练定义好的模型
def train(mnist):
    # 定义输入
    x = tf.placeholder(tf.float32, [None, 784], name='x-input')
    y_ = tf.placeholder(tf.float32, [None, 10], name='y-input')

    # 使用TensorFlow-Slim定义网络结构
    y = lenet5(x)

    # 定义损失函数和训练方法
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=y, labels=tf.argmax(y_, 1))
    loss = tf.reduce_mean(cross_entropy)
    train_op = tf.train.GradientDescentOptimizer(0.01).minimize(loss);

    # 训练过程
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        for i in range(10000):
            xs, ys = mnist.train.next_batch(100)
            _, loss_value = sess.run(
                [train_op, loss], feed_dict={x: xs, y_: ys}
            )
            if i % 1000 == 0:
                print('After %d training steps, loss on training batch is %g.' % (i, loss_value))


def main(agrv=None):
    mnist = input_data.read_data_sets("./path/MNIST_data", one_hot=True)
    train(mnist)


if __name__ == '__main__':
    main()
