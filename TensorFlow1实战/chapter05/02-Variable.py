# -*- coding: utf-8 -*-
# @Date  : 2019/3/23 16:44
# @Software : PyCharm
import tensorflow as tf

a = tf.Variable(0, dtype=tf.float32, name="v")
ema = tf.train.ExponentialMovingAverage(0.99)
maintain_average_op = ema.apply(tf.global_variables())
for var in tf.global_variables():
    print(var.name)


with tf.Session() as sess:
    tf.global_variables_initializer().run()
    sess.run(tf.assign(a, 5))
    sess.run(maintain_average_op)
    print(sess.run([a, ema.average(a)]))