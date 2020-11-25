# -*- coding:utf-8 _*-
"""
#  @Author: DongHao
#  @Date:   2020/11/17 21:36
#  @File:   model.py.py
"""
from tensorflow.keras import layers, Sequential, Model, backend
# from tensorflow.keras.applications.densenet import DenseNet121        # 官方实现


# DenseNet-B
class DenseBlock(layers.Layer):

    def __init__(self, growth_rate):
        super(DenseBlock, self).__init__()

        # ================================================================================================
        bn_axis = 3 if backend.image_data_format() == 'channels_last' else 1
        self.bn1 = layers.BatchNormalization(axis=bn_axis, momentum=0.9, epsilon=1e-5)
        self.relu1 = layers.Activation('relu')
        self.conv1 = layers.Conv2D(filters=growth_rate*4, kernel_size=1, strides=1, use_bias=False, name="block_conv_1x1")
        # ================================================================================================
        self.bn2 = layers.BatchNormalization(axis=bn_axis, momentum=0.9, epsilon=1e-5)
        self.relu2 = layers.Activation('relu')
        self.conv2 = layers.Conv2D(filters=growth_rate, kernel_size=3, strides=1, padding='SAME', use_bias=False, name="block_conv_3x3")

    def __init__(self, growth_rate, **kwargs):
        super(DenseBlock, self).__init__(**kwargs)

        # ================================================================================================
        self.bn1 = layers.BatchNormalization(momentum=0.9, epsilon=1e-5)
        self.relu1 = layers.Activation('relu')
        self.conv1 = layers.Conv2D(filters=growth_rate*4, kernel_size=1, strides=1, use_bias=False)
        # ================================================================================================
        self.bn2 = layers.BatchNormalization(momentum=0.9, epsilon=1e-5)
        self.relu2 = layers.Activation('relu')
        self.conv2 = layers.Conv2D(filters=growth_rate, kernel_size=3, strides=1, padding='SAME', use_bias=False)

    def call(self, inputs, training=False):

        x = self.bn1(inputs, training=training)
        x = self.relu1(x)
        x = self.conv1(x)

        x = self.bn2(x, training=training)
        x = self.relu2(x)
        x = self.conv2(x)

        cn_axis = 3 if backend.image_data_format() == 'channels_last' else 1
        x = layers.Concatenate(axis=cn_axis)([inputs, x])

        return x


# DenseNet-C
class TransitionLayer(layers.Layer):

    def __init__(self, theta=0.5, growth_rate=12):
        super(TransitionLayer, self).__init__()

        # ================================================================================================
        bn_axis = 3 if backend.image_data_format() == 'channels_last' else 1
        self.bn1 = layers.BatchNormalization(axis=bn_axis, momentum=0.9, epsilon=1e-5)
        self.relu1 = layers.Activation('relu')
        self.conv1 = layers.Conv2D(filters=int(growth_rate*theta), kernel_size=1, strides=1, use_bias=False, name="Transit_conv_1x1")
        self.pool1 = layers.AveragePooling2D(pool_size=2, strides=2, padding='SAME')

    def call(self, inputs):

    def __init__(self, theta=0.5, growth_rate=12, **kwargs):
        super(TransitionLayer, self).__init__(**kwargs)

        # ================================================================================================
        self.bn1 = layers.BatchNormalization(momentum=0.9, epsilon=1e-5)
        self.relu1 = layers.Activation('relu')
        self.conv1 = layers.Conv2D(filters=int(growth_rate*theta), kernel_size=1, strides=1, use_bias=False)
        self.pool1 = layers.AveragePooling2D(pool_size=2, strides=2, padding='SAME')

    def call(self, inputs, training=False):

        x = self.bn1(inputs, training=training)
        x = self.relu1(x)
        x = self.conv1(x)
        x = self.pool1(x)

        return x


# DenseNet-BC
class DenseNet(Model):
    def __init__(self, theta, growth_rate, block_num=None, class_num=1000, **kwargs):
        super(DenseNet, self).__init__(**kwargs)
        if block_num is None:
            block_num = [6, 12, 24, 16]

        # ===============================================================================
        self.conv1 = layers.Conv2D(filters=int(growth_rate * 2), kernel_size=7, strides=2, padding='SAME', use_bias=False)
        self.bn1 = layers.BatchNormalization(momentum=0.9, epsilon=1e-5)
        self.relu1 = layers.Activation('relu')
        self.pool1 = layers.MaxPool2D(pool_size=3, strides=2, padding='SAME')
        # ===============================================================================
        self.block1 = self._make_layer(growth_rate, block_num[0])
        self.transition1 = TransitionLayer(theta, growth_rate)
        self.block2 = self._make_layer(growth_rate, block_num[1])
        self.transition2 = TransitionLayer(theta, growth_rate)
        self.block3 = self._make_layer(growth_rate, block_num[2])
        self.transition3 = TransitionLayer(theta, growth_rate)
        self.block4 = self._make_layer(growth_rate, block_num[3])
        # ===============================================================================
        self.bn2 = layers.BatchNormalization(momentum=0.9, epsilon=1e-5)
        self.relu2 = layers.Activation('relu')
        self.pool2 = layers.AveragePooling2D(pool_size=7)
        self.flatten = layers.Flatten()
        self.dense = layers.Dense(class_num, activation='softmax')

    def _make_layer(self, growth_rate, blocks):

        layers_list = [(DenseBlock(growth_rate))]
        for i in range(1, blocks):
            layers_list.append(DenseBlock(growth_rate))

        return Sequential(layers_list)

    def call(self, inputs, training=False, **kwargs):

        x = self.conv1(inputs)              # 224*224*3 => 112*112*24(grow_rate*2)
        # print(x.shape.as_list())
        x = self.bn1(x, training=training)
        x = self.relu1(x)
        x = self.pool1(x)                   # 112*112*24 => 56*56*24
        # print(x.shape.as_list())

        x = self.block1(x)                  # 56*56*24 => 56*56*96
        # print(x.shape.as_list())
        x = self.transition1(x)             # 56*56*96 => 28*28*6
        # print(x.shape.as_list())
        x = self.block2(x)                  # 28*28*6 => 28*28*150
        # print(x.shape.as_list())
        x = self.transition2(x)             # 28*28*150 => 14*14*6
        # print(x.shape.as_list())
        x = self.block3(x)                  # 14*14*6 => 14*14*294
        # print(x.shape.as_list())
        x = self.transition3(x)             # 14*14*294 => 7*7*6
        # print(x.shape.as_list())
        x = self.block4(x)                  # 7*7*6 => 7*7*198
        # print(x.shape.as_list())

        x = self.bn2(x, training=training)
        x = self.relu2(x)
        x = self.pool2(x)                   # 7*7*198 => 1*1*198
        # print(x.shape.as_list())
        x = self.flatten(x)                 # 198
        # print(x.shape.as_list())
        x = self.dense(x)                   # 5
        # print(x.shape.as_list())

        return x


def denseNet121(class_num=1000):
    theta = 0.5
    growth_rate = 12
    block_num = [6, 12, 24, 16]
    return DenseNet(theta, growth_rate, block_num, class_num)
