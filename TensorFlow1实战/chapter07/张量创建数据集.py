# -*- coding: utf-8 -*-
# @Date  : 2019/4/1 9:16
# @Software : PyCharm
import tensorflow as tf

# 从一个数组创建数据集
# input_data = [1, 2, 3, 5, 8]
input_data = tf.placeholder(tf.int32)
dataset = tf.data.Dataset.from_tensor_slices(input_data)
# 在自然语言处理的任务中， 训练数据通常是以每行一条数据的形式存在文本文件中，这时可以用 TextLineDataset 来更方便地读取数据：
# dataset = tf.data.TextLineDataset(input_data)
# 在图像相关任务中，输入数据通常以 TFRecord 形式存储，这时可以用 TFRecordDataset 来读取数据
# dataset = tf.data.TFRecordDataset(input_data)
# 定义一个迭代器用于遍历数据集,因为上面定义的数据集没有用placeholder
# 作为输入参数,所以这里可以使用最简单的one_shot_iterator
iterator = dataset.make_initializable_iterator()
# get_next()返回代表一个输入数据的张量,类似于队列的dequeue()
x = iterator.get_next()
y = x * x
with tf.Session() as sess:
    # tf.global_variables_initializer().run()

    sess.run(iterator.initializer, feed_dict={input_data: [9]})

    while True:
        try:
            print(sess.run(y))
        except tf.errors.OutOfRangeError:
            break