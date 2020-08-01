# -*- coding: utf-8 -*-
# @Date  : 2019/4/14 11:01
# @Software : PyCharm
import tflearn
from tflearn.layers.core import input_data, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression
import tflearn.datasets.mnist as mnist

# 读取mnist数据
trainX, trainY, testX, textY = mnist.load_data(data_dir='./path/MNIST_data', one_hot=True)

# 将图像数据reshape成卷积神经网络输入的格式
trainX = trainX.reshape([-1, 28, 28, 1])
testX = testX.reshape([-1, 28, 28, 1])

# 构建神经网络,这个过程和TensorFlow-Slim比较类似,input_data定义了一个placeholder来接入输入数据
net =  input_data(shape=[None,28,28,1],name='input')
# 通过TFLearn封装好的API定义一个深度为5,过滤器为5x5,激活函数为ReLu的卷积层
net = conv_2d(net,32,5,activation='relu')
# 定义一个过滤器为2x2的最大池化层
net = max_pool_2d(net,2)
net = conv_2d(net,64,5,activation='relu')
net = max_pool_2d(net,2)
net = fully_connected(net,500,activation='relu')
net = fully_connected(net,10,activation='relu')

# 使用TFLearn封装好的函数定义学习任务指定优化器为sgd,学习率为0.01,损失函数为交叉熵
net = regression(net, optimizer='sgd',learning_rate=0.01,loss='categorical_crossentropy')

# 通过定义的网络结构训练模型,并在指定的验证数据上验证模型的效果,TFLearn将模型的训练过程封装到一个类中,这样
# 可以减少非常多的冗余代码
model = tflearn.DNN(net, tensorboard_verbose=0)
model.fit(trainX,trainY,n_epoch=20,validation_set=([testX,textY]),show_metric=True)