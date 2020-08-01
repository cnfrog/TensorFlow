# -*- coding: utf-8 -*-
# @Date  : 2019/3/30 11:40
# @Software : PyCharm
import tensorflow as tf


# 创建一个先进先出队列
q = tf.FIFOQueue(2, "int32")
# 使用enqueue_many函数来初始化队列中的元素.和变量初始化类似,在使用队列之前需要明确的调用的这个初始化过程
init = q.enqueue_many(([0, 10],))

# 使用Dequeue函数将队列中的第一个元素出队列.这个元素的值将存在变量x中
x = q.dequeue()
y = x+1
q_inc = q.enqueue([y])

with tf.Session() as sess:
    # 运行初始化队列的操作
    init.run()

    for _ in range(5):
        v, _ = sess.run([x, q_inc])
        print(v)

