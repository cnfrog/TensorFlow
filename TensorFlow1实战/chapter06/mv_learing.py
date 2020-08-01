# -*- coding: utf-8 -*-
# @Date  : 2019-03-28 19:56
# @Software : PyCharm
import tensorflow as tf
import glob
import os.path
import numpy as np
from tensorflow.python.platform import gfile
import tensorflow.contrib.slim as slim

# 加载通过TensorFlow-Slim定义好的inception_v3模型
import tensorflow.contrib.slim.python.slim.nets.inception_v3 as inception_v3

# 处理好之后的数据文件
INPUT_DATA = './path/to/flower_processed_data.npy'
# 保存训练好的模型的路径,这样可以将新数据训练得到的完整模型保存
# 下来,如果计算资源充足,还可以在训练完最后的全连接层之后再训练所有
# 网络层,这样可以使得新模型更加贴近新数据
TRAIN_FILE = './path/to/save_model'
# 谷歌提供的训练好的模型文件地址
CKPT_FILE = './path/to/incaption_v3.ckpt'
# 定义训练中使用的参数
LEARNING_RATE = 0.0001
STEPS = 300
BATCH = 32
N_CLASSES = 5

# 不需要从谷歌训练好的模型中加载参数,这里就是最后的全连接层,因为在
# 新的问题中要重新训练这一层中的参数,这里给出的是参数的前缀
CHECKPOINT_EXCLUDE_SCOPES = 'InceptionV3/Logits,InceptionV3/AuxLogits'
# 需要训练的网络层参数名称,在fine-tuning的过程中就是最后的全连接层
# 这里给出的是参数的前缀
TRAIN_SCOPES = 'InceptionV3/Logits,InceptionV3/AuxLogits'


# 获取所有需要从谷歌训练好的模型中加载的参数
def get_tuned_variables():
    exclusions = [scope.strip() for scope in CHECKPOINT_EXCLUDE_SCOPES.split(',')]

    variable_to_restore = []
    # 枚举inception-v3模型中所有的参数,然后判断是否需要从加载列表中移除
    for var in slim.get_model_variables():
        excluded = False
        for exclusion in exclusions:
            if var.op.name.startswith(exclusion):
                excluded = True
                break
        if not excluded:
            variable_to_restore.append(var)
    return variable_to_restore


# 获取所有需要训练的变量列表
def get_trainable_variables():
    scopes = [scope.strip() for scope in TRAIN_SCOPES.strip(',')]
    variables_to_train = []
    # 枚举所有需要训练的参数前缀,并通过这些前缀找到所有的参数
    for scope in scopes:
        variables = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, scope)
        variables_to_train.append(variables)
    return variables_to_train


def get_data():
    try:
        with open(INPUT_DATA, 'rb') as f:
            return np.load(f)
    except EOFError as e:  # 捕获异常EOFError 后返回None
        print(str(e))
        return None


def main(argv=None):
    # 加载预处理好的数据
    processed_data = get_data()
    training_images = processed_data[0]
    n_training_example = len(training_images)
    training_labels = processed_data[1]
    validation_images = processed_data[2]
    validation_labels = processed_data[3]
    testing_images = processed_data[4]
    testing_labels = processed_data[5]

    print("%d training examples,%d validation examples and %d testing images" % (n_training_example,
                                                                                 len(validation_images),
                                                                                 len(testing_images)))
    # 定义Incepti-v3的输入,images为输入图片,labels为每一张图片对应的标签
    images = tf.placeholder(tf.float32, [None, 299, 299, 3], name='input_images')
    labels = tf.placeholder(tf.float32, [None], name='labels')

    # 定义inception-v3模型.因为谷歌给出的只有模型参数取值,所以这里需要在这个代码中定义inception-v3的模型结构.虽然理论上需要区分
    # 训练和测试中使用的模型,也就是说在测试时应该使用is_training=False,但是因为预先训练好的inception-v3
    # 模型中使用的batch normalization参数与新的数据会有差异,导致结果很差,所以这里直接使用同一个模型来进行测试
    with slim.arg_scope(inception_v3.inception_v3_arg_scope()):
        logits, _ = inception_v3.inception_v3(images, num_classes=N_CLASSES)
    # 获取需要训练的变量
    trainable_variables = get_trainable_variables()
    # 定义交叉熵损失,注意在模型定义的时候已经将正则化损失加入损失集合中
    tf.losses.softmax_cross_entropy(tf.one_hot(labels, N_CLASSES), logits, weights=1.0)

    # 定义训练过程,这里minimize的过程中指定了需要优化的变量结合
    train_step = tf.train.RMSPropOptimizer(LEARNING_RATE).minimize(tf.losses.get_total_loss)

    # 计算正确率
    with tf.name_scope('evaluation'):
        corrent_prediction = tf.equal(tf.argmax(logits, 1), labels)
        evaluation_step = tf.reduce_mean(tf.cast(corrent_prediction, tf.float32))

    # 定义加载模型的函数
    load_fn = slim.assign_from_checkpoint_fn(CKPT_FILE, get_tuned_variables(), ignore_missing_vars=True)

    # 定义保存新的训练好的模型的函数
    saver = tf.train.Saver()
    with tf.Session() as sess:
        tf.global_variables_initializer().run()

        print("Load tuned variables from %s" % CKPT_FILE)
        load_fn(sess)

        start = 0
        end = BATCH
        for i in range(STEPS):
            sess.run(train_step, feed_dict={images: training_images[start:end], labels: training_labels[start:end]})

            if i % 30 == 0 or i + 1 == STEPS:
                saver.save(sess, TRAIN_FILE, global_step=1)
                validation_accuracy = sess.run(evaluation_step,
                                               feed_dict={images: validation_images, labels: validation_labels})
                print("step %d: Validation accuracy = %.1f%%" % (i, validation_accuracy * 100.0))

            start = end
            if start == n_training_example:
                start = 0
            end = start + BATCH
            if end > n_training_example:
                end = n_training_example

        test_accuracy = sess.run(evaluation_step,
                                 feed_dict={images: testing_images, labels: testing_labels})
        print("Final test accuracy = %.1f%%" % (test_accuracy * 100.0))


if __name__ == '__main__':
    tf.app.run()
