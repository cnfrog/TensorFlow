# -*- coding: utf-8 -*-
# @Date  : 2019/3/29 15:29
# @Software : PyCharm
import tensorflow as tf
import matplotlib.pyplot as plt

# 读取图像的原始数据
image_raw_data = tf.read_file("./path/pic/b.jpg")

def image_show(img_data):
    plt.imshow(img_data.eval())
    plt.show()

with tf.Session() as sess:
    # 对图像进行jpeg的格式编码从而得到图像对应的三维矩阵
    # TensorFlow还提供tf.image.decode_png函数对png格式的图像进行编码
    # 解码之后的结果为一个张量,在使用它之前需要明确调用运行的过程
    img_data = tf.image.decode_jpeg(image_raw_data)


    encoded_image = tf.image.encode_jpeg(img_data)
    with tf.gfile.GFile("./path/to/output.jpg", 'wb') as f:
        f.write(encoded_image.eval())

    # 加载原始图像，定义会话等过程和图像编码处理中代码一致， 假设
    # img_data是己经解码的图像。
    # 首先将图片数据转化为实数类型。这一步将 0 - 255的像素值转化为0.0 - 1.0范围内的实数。
    # 大多数图像处理API支持整数和实数类型的输入。如果输入是整数类础，这些API
    # 会在内部将输入转化为实数后处理， 再将输出转化为整数.
    # 如果有多个处理步骤，在整数和实数之间的反复转化将导致精度损失，因此推荐在图像处理前将其转化为实数类型。
    # 下面的样例子将略去这一步骤，假设img_data是经过类型转化的图像。
    img_data = tf.image.convert_image_dtype(img_data, dtype = tf.float32)
    # 通过tf.image.resize_images函数调整图像的大小。这个函数第一个参数为原始图像，
    # 第二个和第三个参数为调整后图像的大小， method参数给出了调整图像大小的算法。
    # 注意，如果输入数据是unit8格式，那么输出将是 0～255之内的实数，不方便后续处理。
    # 本书建议在调整图像大小前先转化为实数类型。
    '''
    method 
    0 双线性插值法
    1 最近邻居法
    2 双三次插值法
    3 面积插值法
    
    '''
    resized = tf.image.resize_images(img_data, [300, 300], method=0)
    #通过 pyplot可视化的过程和图像编码处理中给出的代码一致，在以下代码中也将略去
    image_show(resized)


    # 通过tf.image.resize_image_with_crop_or_pad函数调整图像的大小。这个函数的
    # 第一个个参数为原始图像，后面两个参数是调整后的目标图像大小。如果原始图像的尺寸大于目标
    # 图像，那么这个的函数会自动截取原始图像中居中的部分
    # 如果目标图像大于原始图像，这个函数会在原始图像的四周填充全0背景
    croped = tf.image.resize_image_with_crop_or_pad(img_data, 1000, 1000)
    padded = tf.image.resize_image_with_crop_or_pad(img_data, 5000, 5000)
    image_show(croped)
    image_show(padded)

    # 通过tf.image.central_crop函数可以按比例裁剪图像。这个函数的第一个参数为原始图
    # 像， 第二个为调整比例，这个比例需要是 －个（0,1］的实数
    central_cropped = tf.image.central_crop(img_data, 0.5)
    image_show(central_cropped)

    # 将图像上下翻转
    flipped = tf.image.flip_up_down(img_data)
    # 将图像左右翻转
    flipped = tf.image.flip_left_right(img_data)
    # 将图像沿对角线翻转
    transposed = tf.image.transpose_image(img_data)
    image_show(transposed)

    # 以50%概率上下翻转图像
    flipped = tf.image.random_flip_up_down(img_data)
    # 以50%概率左右翻转图像
    flipped = tf.image.random_flip_left_right(img_data)


    # 将图像的亮度－0.5，
    adjusted = tf.image.adjust_brightness(img_data, -0.5)
    # 色彩调整的API可能导致像素的实数值超出0.0 - 1.0的范固，因此在输出最终图像前需要
    # 将其值截断在0.0 - 1.0范围区间，否则不仅图像无法正常可视化，以此为输入的神经网络
    # 的训练质量也可能受到影响。
    # 如果对图像进行多项处理操作，那么这一截断过程应当在所有处理完成后进行。举例而言，
    # 假如对图像依次提高亮度和减少对比度，那么第二个操作可能将第一个操作生成的部分
    # 过亮的像素拉回到不超过1.0的范围内，因此在第一个操作后不应该立即截断。
    # 下面的作例假设截断操作在最终可视化图像前进行。
    adjusted= tf.clip_by_value(adjusted, 0.0, 1.0)
    # 将图像的亮度＋0.5，
    adjusted = tf.image.adjust_brightness(img_data, 0.5)
    #在［－max_delta, max_delta)的范围随机调整图像的亮度。
    # adjusted = tf.image.random_brightness(image, max_delta)
    image_show(adjusted)

    # 将图像的对比度减少到0.5倍
    adjusted = tf.image.adjust_contrast(img_data, 0.5)
    # 将图像的对比度增加5倍
    adjusted = tf.image.adjust_contrast(img_data, 5)
    # 在 ［ lower, upper）的范围随机调整图的对比度。
    # adjusted = tf.image.random_contrast(image, lower, upper)
    image_show(adjusted)

    # 下面命令将色相加0.1
    adjusted = tf.image.adjust_hue(img_data, 0.1)
    image_show(adjusted)
    # 在［－max_delta, max_delta）的范围内随机调整图像的色相。
    # max delta 的取值在（ 0, 0 . 5 ］之间。
    # adjusted= tf.image.random_hue(image, max_delta)

    # 将图像的饱和度-5
    adjusted = tf.image.adjust_saturation(img_data, -5)
    # 在[lower, upper]的范围内随机调整图像的饱和度
    # adjusted = tf.image.random_saturation(image, lower, upper)

    # 将代表一张图像的三维矩阵中的数字均值变为0，方差变为1
    adjusted = tf.image.per_image_standardization(img_data)
    image_show(adjusted)


    # tf.image.draw_bounding_boxes 函数要求图像矩阵中的数字为实数，所以需要先将
    # 图像矩阵转化为实数类型。 tf.image.draw_bounding_boxes函数图像的输入是一个
    # bacth的数据，也就是多张图像组成的四维矩阵，所以需要将解码之后的图像矩阵加一维。
    # 给出每一张图像的所有标注框。一个标注框有 4 个数字，分别代农 ［Ymin Xmin Ymax Xmax]。
    # 注意这里给出的数字都是图像的相对位置。比如在 180 × 267 的阁像中，
    # (0.35, 0 . 47, 0 . 5, 0 . 56］代表了从（ 63, 125）到（ 90, 150 ）的图像。
    boxes = tf.constant([[[0.05, 0.05, 0.9, 0.7], [0.35, 0.47, 0.5, 0.56]]])
    # 可以通过提供标注框的方式来告诉随机截取图像的算法哪些部分是“有信息量”的。
    # min_object_covered=0.4 表示截取部分至少包含某个标注框40%的内容。
    begin, size, bbox_for_draw = tf.image.sample_distorted_bounding_box(
        tf.shape(img_data), bounding_boxes=boxes, min_object_covered=0.4)

    # 截取后的图片
    distorted_image = tf.slice(img_data, begin, size)

    # 在原图上用标注框画出截取的范围。由于原图的分辨率较大（2673x1797)，生成的标注框
    # 在Jupyter Notebook上通常因边框过细而无法分辨，这里为了演示方便先缩小分辨率。
    image_small = tf.image.resize_images(img_data, [180, 267], method=0)
    batchced_img = tf.expand_dims(image_small, 0)
    image_with_box = tf.image.draw_bounding_boxes(batchced_img, bbox_for_draw)
    print(image_with_box.eval())
    image_show(image_with_box[0])


