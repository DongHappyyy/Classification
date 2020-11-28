# -*- coding:utf-8 _*-
"""
#  @Author: DongHao
#  @Date:   2020/11/26 9:59
#  @File:   GoogLeNet.py
"""
import tensorflow as tf
from tensorflow.keras import layers, Model, Sequential, backend, activations


class Inception(layers.Layer):

    def __init__(self, conv1_num, conv1_3x3, conv3_num, conv1_5x5, conv5_num, conv1_pool, Name, **kwargs):
        super(Inception, self).__init__(**kwargs)
        self.conv1_num = conv1_num
        self.conv1_3x3 = conv1_3x3
        self.conv3_num = conv3_num
        self.conv1_5x5 = conv1_5x5
        self.conv5_num = conv5_num
        self.conv1_pool = conv1_pool
        self.inceptionName = Name
        
    def build(self, input_shape):
        self.branch1 = Sequential([
            layers.Conv2D(filters=self.conv1_num, kernel_size=1, use_bias=False, name="branch1_Conv_1x1"),
            layers.BatchNormalization(momentum=0.9, epsilon=1e-5, name="branch1_BatchNormalization"),
            layers.ReLU()
        ], name=str(self.inceptionName)+"_branch1")
        self.branch2 = Sequential([
            layers.Conv2D(filters=self.conv1_3x3, kernel_size=1, use_bias=False, name="branch2_Conv_1x1"),
            layers.BatchNormalization(momentum=0.9, epsilon=1e-5, name="branch2_BatchNormalization1"),
            layers.ReLU(),
            layers.Conv2D(filters=self.conv3_num, kernel_size=3, strides=1, padding='SAME', use_bias=False, name="branch2_Conv_3x3"),
            layers.BatchNormalization(momentum=0.9, epsilon=1e-5, name="branch2_BatchNormalization2"),
            layers.ReLU()
        ], name=str(self.inceptionName)+"_branch2")
        self.branch3 = Sequential([
            layers.Conv2D(filters=self.conv1_5x5, kernel_size=1, use_bias=False, name="branch3_Conv_1x1"),
            layers.BatchNormalization(momentum=0.9, epsilon=1e-5, name="branch3_BatchNormalization1"),
            layers.ReLU(),
            layers.Conv2D(filters=self.conv5_num, kernel_size=5, strides=1, padding='SAME', use_bias=False, name="branch3_Conv_5x5"),
            layers.BatchNormalization(momentum=0.9, epsilon=1e-5, name="branch3_BatchNormalization2"),
            layers.ReLU()
        ], name=str(self.inceptionName)+"_branch3")
        self.branch4 = Sequential([
            layers.MaxPool2D(pool_size=3, strides=1, padding='SAME', name="branch4_MaxPool"),
            layers.Conv2D(filters=self.conv1_pool, kernel_size=1, use_bias=False, name="branch4_Conv_1x1"),
            layers.BatchNormalization(momentum=0.9, epsilon=1e-5, name="branch4_BatchNormalization2"),
            layers.ReLU()
        ], name=str(self.inceptionName)+"_branch4")
        
        super(Inception, self).build(input_shape)

    def call(self, inputs, **kwargs):
        branch1 = self.branch1(inputs)
        branch2 = self.branch2(inputs)
        branch3 = self.branch3(inputs)
        branch4 = self.branch4(inputs)

        b_axis = 3 if backend.image_data_format() == "channels_last" else 1
        outputs = layers.Concatenate(axis=b_axis)([branch1, branch2, branch3, branch4])

        return outputs


class AuxiliaryClassifier(layers.Layer):

    def __init__(self, class_num, **kwargs):
        super(AuxiliaryClassifier, self).__init__(**kwargs)
        self.class_num = class_num

    def build(self, input_shape):
        self.maxpool = layers.MaxPool2D(pool_size=5, strides=3, name="auxiliary_classifier_MaxPool")
        self.conv = layers.Conv2D(filters=128, kernel_size=1, strides=1, use_bias=False, name="auxCla_Conv1")
        self.bn = layers.BatchNormalization(momentum=0.9, epsilon=1e-5, name="auxiliary_classifier_BatchNormalization")
        self.act = activations.relu
        self.flatten = layers.Flatten(name="Flatten_AUX")
        self.dense1 = layers.Dense(units=1024, activation='relu', name="auxiliary_classifier_Dense1")
        self.drop = layers.Dropout(0.5)
        self.dense2 = layers.Dense(units=self.class_num, activation='softmax', name="auxiliary_classifier_Dense2")

        super(AuxiliaryClassifier, self).build(input_shape)

    def call(self, inputs, training=False, **kwargs):
        x = self.maxpool(inputs)
        x = self.conv(x)
        x = self.bn(x, training=training)
        x = self.act(x)

        x = self.flatten(x)
        x = self.dense1(x)
        x = self.drop(x)
        x = self.dense2(x)

        return x


class GoogLeNet(Model):

    def __init__(self, class_num, **kwargs):
        super(GoogLeNet, self).__init__(**kwargs)
        self.class_num = class_num

    def build(self, input_shape):
        self.conv1 = layers.Conv2D(filters=64, kernel_size=7, strides=2, padding='SAME', use_bias=False, name="Conv_7x7")
        self.bn1 = layers.BatchNormalization(momentum=0.9, epsilon=1e-5, name="BatchNormalization_1")
        self.act1 = activations.relu

        self.maxpool1 = layers.MaxPool2D(pool_size=3, strides=2, padding='SAME', name="MAXPooling_1")

        self.conv2 = layers.Conv2D(filters=64, kernel_size=1, strides=1, padding='SAME', use_bias=False, name="Conv_1x1")
        self.bn2 = layers.BatchNormalization(momentum=0.9, epsilon=1e-5, name="BatchNormalization_2")
        self.act2 = activations.relu

        self.conv3 = layers.Conv2D(filters=192, kernel_size=3, strides=1, padding='SAME', use_bias=False, name="Conv_3x3")
        self.bn3 = layers.BatchNormalization(momentum=0.9, epsilon=1e-5, name="BatchNormalization_3")
        self.act3 = activations.relu

        self.maxpool2 = layers.MaxPool2D(pool_size=3, strides=2, padding='SAME', name="MAXPooling_2")

        self.inception1 = Inception(64, 96, 128, 16, 32, 32, "Inception1")
        self.inception2 = Inception(128, 128, 192, 32, 96, 64, "Inception2")

        self.maxpool3 = layers.MaxPool2D(pool_size=3, strides=2, padding='SAME', name="MAXPooling_3")

        self.inception3 = Inception(192, 96, 208, 16, 48, 64, "Inception3")
        self.auxiliary_classifier1 = AuxiliaryClassifier(self.class_num)

        self.inception4 = Inception(160, 112, 224, 24, 64, 64, "Inception4")
        self.inception5 = Inception(128, 128, 256, 24, 64, 64, "Inception5")
        self.inception6 = Inception(112, 144, 288, 32, 64, 64, "Inception6")
        self.auxiliary_classifier2 = AuxiliaryClassifier(self.class_num)

        self.inception7 = Inception(256, 160, 320, 32, 128, 128, "Inception7")

        self.maxpool4 = layers.MaxPool2D(pool_size=3, strides=2, padding='SAME', name="MAXPooling_4")

        self.inception8 = Inception(256, 160, 320, 32, 128, 128, "Inception8")
        self.inception9 = Inception(384, 192, 384, 48, 128, 128, "Inception9")

        self.avepool = layers.AvgPool2D(pool_size=7, strides=1, name="AveragePooling_1")
        self.flatten = layers.Flatten(name="Flatten")
        self.drop = layers.Dropout(0.4)
        self.dense = layers.Dense(self.class_num, activation='softmax', name="Dense_1")

        super(GoogLeNet, self).build(input_shape)

    def call(self, inputs, training=False, **kwargs):
        x = self.conv1(inputs)                      # 224x224x3 => 112x112x64
        # print(x.shape.as_list())
        x = self.bn1(x, training=training)
        x = self.act1(x)

        x = self.maxpool1(x)                        # 112x112x64 => 56x56x64
        # print(x.shape.as_list())

        x = self.conv2(x)                           # 56x56x64 => 56x56x64
        # print(x.shape.as_list())
        x = self.bn2(x, training=training)
        x = self.act2(x)

        x = self.conv3(x)                           # 56x56x64 => 56x56x192
        # print(x.shape.as_list())
        x = self.bn3(x, training=training)
        x = self.act3(x)

        x = self.maxpool2(x)                        # 56x56x192 => 28x28x192
        # print(x.shape.as_list())

        x = self.inception1(x)                      # 28x28x192 => 28x28x256
        # print(x.shape.as_list())
        x = self.inception2(x)                      # 28x28x192 => 28x28x480
        # print(x.shape.as_list())

        x = self.maxpool3(x)                        # 28x28x480 => 14x14x480
        # print(x.shape.as_list())

        x = self.inception3(x)                      # 14x14x480 => 14x14x512
        # print(x.shape.as_list())
        auxiliary_classifier_1 = self.auxiliary_classifier1(x)  # => 5
        # print(auxiliary_classifier_1.shape.as_list())

        x = self.inception4(x)                      # 14x14x512 => 14x14x512
        # print(x.shape.as_list())
        x = self.inception5(x)                      # 14x14x512 => 14x14x512
        # print(x.shape.as_list())
        x = self.inception6(x)                      # 14x14x512 => 14x14x528
        # print(x.shape.as_list())
        auxiliary_classifier_2 = self.auxiliary_classifier2(x)  # => 5
        # print(auxiliary_classifier_2.shape.as_list())

        x = self.inception7(x)                      # 14x14x528 => 14x14x832
        # print(x.shape.as_list())

        x = self.maxpool4(x)                        # 14x14x832 => 7x7x832
        # print(x.shape.as_list())

        x = self.inception8(x)                      # 7x7x832 => 7x7x832
        # print(x.shape.as_list())
        x = self.inception9(x)                      # 7x7x832 => 7x7x1024
        # print(x.shape.as_list())

        x = self.avepool(x)                         # 7x7x1024 => 1x1x1024
        # print(x.shape.as_list())
        x = self.flatten(x)                         # 1x1x1024 => 1024
        # print(x.shape.as_list())
        x = self.drop(x)
        main_classifier = self.dense(x)             # 1024 => 5
        # print(main_classifier.shape.as_list())

        return auxiliary_classifier_1, auxiliary_classifier_2, main_classifier
