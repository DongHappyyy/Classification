# -*- coding:utf-8 _*-
"""
#  @Author: DongHao
#  @Date:   2020/11/6 13:38
#  @File:   model.py
# MobileNet V2
"""
from tensorflow.keras import Model, layers, Sequential, activations


class Bottleneck(layers.Layer):

    def __init__(self, output, strides=1, shortcut=False, **kwargs):
        super(Bottleneck, self).__init__(**kwargs)

        # ---------------------------------------------------------------------------------------------------------
        # PW, 只改变通道数，不改变宽度和高度，所以 strides=1
        self.conv1 = layers.Conv2D(filters=output, kernel_size=1, strides=1, padding='SAME', use_bias=False)
        self.bn1 = layers.BatchNormalization(momentum=0.9, epsilon=1e-5)
        self.relu1 = layers.ReLU(max_value=6)
        # ---------------------------------------------------------------------------------------------------------
        # DW, 只改变宽度和高度，不改变通道数
        self.conv2 = layers.DepthwiseConv2D(kernel_size=3, strides=strides, padding='SAME', use_bias=False)
        self.bn2 = layers.BatchNormalization(momentum=0.9, epsilon=1e-5)
        self.relu2 = layers.ReLU(max_value=6)
        # ---------------------------------------------------------------------------------------------------------
        # PW, 只改变通道数，不改变宽度和高度，所以 strides=1
        self.conv3 = layers.Conv2D(filters=output, kernel_size=1, strides=1, padding='SAME', use_bias=False)
        self.bn3 = layers.BatchNormalization(momentum=0.9, epsilon=1e-5)
        self.linear = activations.linear
        # ---------------------------------------------------------------------------------------------------------
        self.shortcut = shortcut
        self.add = layers.Add()

    def call(self, inputs, training=False, **kwargs):
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.bn2(x, training=training)
        x = self.relu2(x)

        x = self.conv3(x)
        x = self.bn3(x, training=training)
        x = self.linear(x)

        if self.shortcut is True:
            x = self.add([inputs, x])

        return x


class MobileNet(Model):

    def __init__(self, block, block_num, num_classes=1000, **kwargs):
        super(MobileNet, self).__init__(**kwargs)

        # ---------------------------------------------------------------------------------------------------------
        self.conv1 = layers.Conv2D(filters=32, kernel_size=3, strides=2, padding='SAME', use_bias=False)
        self.bn1 = layers.BatchNormalization(momentum=0.9, epsilon=1e-5)
        self.relu1 = layers.ReLU(max_value=6)
        # ---------------------------------------------------------------------------------------------------------
        self.bottleneck1 = self.make_layer(block, block_num[0], 16, strides=1, shortcut=False)
        self.bottleneck2 = self.make_layer(block, block_num[1], 24, strides=2, shortcut=False)
        self.bottleneck3 = self.make_layer(block, block_num[2], 32, strides=2, shortcut=False)
        self.bottleneck4 = self.make_layer(block, block_num[3], 64, strides=2, shortcut=False)
        self.bottleneck5 = self.make_layer(block, block_num[4], 96, strides=1, shortcut=False)
        self.bottleneck6 = self.make_layer(block, block_num[5], 160, strides=2, shortcut=False)
        self.bottleneck7 = self.make_layer(block, block_num[6], 320, strides=1, shortcut=False)
        # ---------------------------------------------------------------------------------------------------------
        self.conv2 = layers.Conv2D(filters=1280, kernel_size=1, strides=1, padding='SAME', use_bias=False)
        self.bn2 = layers.BatchNormalization(momentum=0.9, epsilon=1e-5)
        self.relu2 = layers.ReLU(max_value=6)
        # ---------------------------------------------------------------------------------------------------------
        self.avgpool = layers.GlobalAvgPool2D()     # pool + flatten
        self.fc = layers.Dense(num_classes, activation='softmax')

    def make_layer(self, block, block_num, channel, strides, shortcut=False):
        layer_list = [(block(channel, strides=strides, shortcut=shortcut))]
        for i in range(1, block_num):
            layer_list.append(block(channel, strides=1, shortcut=True))
        return Sequential(layer_list)

    def call(self, inputs, training=False, **kwargs):
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = self.relu1(x)

        x = self.bottleneck1(x, training=training)
        x = self.bottleneck2(x, training=training)
        x = self.bottleneck3(x, training=training)
        x = self.bottleneck4(x, training=training)
        x = self.bottleneck5(x, training=training)
        x = self.bottleneck6(x, training=training)
        x = self.bottleneck7(x, training=training)

        x = self.conv2(x)
        x = self.bn2(x, training=training)
        x = self.relu2(x)

        x = self.avgpool(x)
        x = self.fc(x)

        return x


def mobilenet(num_classes=1000):
    blocks_num = [1, 2, 3, 4, 3, 3, 1]
    return MobileNet(Bottleneck, blocks_num, num_classes)

# dw卷积如何实现的     DepthwiseConv2D
# 线性激活如何实现的     activations.linear(x)
# shutcut如何实现的      shortcut=False， 每个规模一样的bottleneck除第一个外其余都有shutcut，每个不同bottleneck的第一个，都没有shutcut，因为前后通道都会改变（在此例中）
#                                       每个不同bottleneck的第一个中，第一个pw实现通道改变，后面所有pw都只是维持，Dw进行改变大小，后面的Dw都是维持
