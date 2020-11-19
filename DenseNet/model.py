# -*- coding:utf-8 _*-
"""
#  @Author: DongHao
#  @Date:   2020/11/17 21:36
#  @File:   model.py.py
"""
from tensorflow.keras import layers, Sequential, Model, activations


# DenseNet-B
class DenseBlock(layers.Layer):
    def __init__(self, K, **kwargs):
        super(DenseBlock, self).__init__(**kwargs)

        # ===============================================================================
        self.bn1 = layers.BatchNormalization(momentum=0.9)
        self.relu1 = layers.ReLU()
        self.conv1 = layers.Conv2D(filters=K*4, kernel_size=1, strides=1, padding='SAME', use_bias=False)
        # ===============================================================================
        self.bn1 = layers.BatchNormalization(momentum=0.9)
        self.relu1 = layers.ReLU()
        self.conv1 = layers.Conv2D(filters=K, kernel_size=3, strides=1, padding='SAME', use_bias=False)


# DenseNet-C
class TransitionLayer(layers.Layer):
    def __init__(self, theta, K, **kwargs):
        super(TransitionLayer, self).__init__(**kwargs)

        self.bn1 = layers.BatchNormalization(momentum=0.9)
        self.relu1 = layers.ReLU()
        self.conv1 = layers.Conv2D(filters=int(K*theta), kernel_size=1, strides=1, use_bias=False)
        self.pool1 = layers.AvgPool2D(pool_size=2, strides=2)

    def call(self, inputs, **kwargs):

        x = self.bn1(inputs)
        x = self.relu1(x)
        x = self.conv1(x)
        x = self.pool1(x)

        return x


# DenseNet-BC
class DenseNet(Model):
    def __init__(self, theta, K, block_num=None, class_num=1000, **kwargs):
        super(DenseNet, self).__init__(**kwargs)
        if block_num is None:
            block_num = [6, 12, 24, 16]

        # ===============================================================================
        self.bn1 = layers.BatchNormalization(momentum=0.9)
        self.relu1 = layers.ReLU()
        self.conv1 = layers.Conv2D(filters=int(K * 2), kernel_size=7, strides=2, padding='SAME', use_bias=False)
        self.pool1 = layers.MaxPool2D(pool_size=3, strides=2)

        # ===============================================================================
        self.bn1 = layers.BatchNormalization(momentum=0.9)
        self.relu1 = layers.ReLU()
        self.conv1 = layers.Conv2D(filters=int(K), kernel_size=7, strides=2, padding='SAME', use_bias=False)
        self.pool2 = layers.GlobalAvgPool2D(pool_size=7)
        self.dense1 = layers.Dense(class_num)


def denseNet121():
    growth_rate = 12
    theta = 1
    block_num = [6, 12, 24, 16]
    class_num = 5
    return DenseNet(theta, growth_rate, block_num, class_num)
