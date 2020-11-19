# -*- coding:utf-8 _*-
"""
#  @Author: DongHao
#  @Date:   2020/11/6 21:41
#  @File:   model.py
"""
from tensorflow.keras import layers, Model


class VGG16(Model):

    def __init__(self, class_num=1000, **kwargs):
        super(VGG16, self).__init__(**kwargs)

        # -----------------------------------------------------------------------------------------
        self.conv1 = layers.Conv2D(filters=64, kernel_size=3, strides=1, padding='SAME', use_bias=False)
        self.bn1 = layers.BatchNormalization(momentum=0.9, epsilon=1e-5)
        self.relu1 = layers.ReLU()
        self.conv2 = layers.Conv2D(filters=64, kernel_size=3, strides=1, padding='SAME', use_bias=False)
        self.bn2 = layers.BatchNormalization(momentum=0.9, epsilon=1e-5)
        self.relu2 = layers.ReLU()
        self.pooll = layers.MaxPool2D(pool_size=2, strides=2, padding='same')
        self.drop1 = layers.Dropout(0.2)  # dropout层
        # -----------------------------------------------------------------------------------------
        self.conv3 = layers.Conv2D(filters=128, kernel_size=3, strides=1, padding='SAME', use_bias=False)
        self.bn3 = layers.BatchNormalization(momentum=0.9, epsilon=1e-5)
        self.relu3 = layers.ReLU()
        self.conv4 = layers.Conv2D(filters=128, kernel_size=3, strides=1, padding='SAME', use_bias=False)
        self.bn4 = layers.BatchNormalization(momentum=0.9, epsilon=1e-5)
        self.relu4 = layers.ReLU()
        self.pool2 = layers.MaxPool2D(pool_size=2, strides=2, padding='same')
        self.drop2 = layers.Dropout(0.2)  # dropout层
        # -----------------------------------------------------------------------------------------
        self.conv5 = layers.Conv2D(filters=256, kernel_size=3, strides=1, padding='SAME', use_bias=False)
        self.bn5 = layers.BatchNormalization(momentum=0.9, epsilon=1e-5)
        self.relu5 = layers.ReLU()
        self.conv6 = layers.Conv2D(filters=256, kernel_size=3, strides=1, padding='SAME', use_bias=False)
        self.bn6 = layers.BatchNormalization(momentum=0.9, epsilon=1e-5)
        self.relu6 = layers.ReLU()
        self.conv7 = layers.Conv2D(filters=256, kernel_size=3, strides=1, padding='SAME', use_bias=False)
        self.bn7 = layers.BatchNormalization(momentum=0.9, epsilon=1e-5)
        self.relu7 = layers.ReLU()
        self.pool3 = layers.MaxPool2D(pool_size=2, strides=2, padding='SAME')
        self.drop3 = layers.Dropout(0.2)  # dropout层
        # -----------------------------------------------------------------------------------------
        self.conv8 = layers.Conv2D(filters=512, kernel_size=3, strides=1, padding='SAME', use_bias=False)
        self.bn8 = layers.BatchNormalization(momentum=0.9, epsilon=1e-5)
        self.relu8 = layers.ReLU()
        self.conv9 = layers.Conv2D(filters=512, kernel_size=3, strides=1, padding='SAME', use_bias=False)
        self.bn9 = layers.BatchNormalization(momentum=0.9, epsilon=1e-5)
        self.relu9 = layers.ReLU()
        self.conv10 = layers.Conv2D(filters=512, kernel_size=3, strides=1, padding='SAME', use_bias=False)
        self.bn10 = layers.BatchNormalization(momentum=0.9, epsilon=1e-5)
        self.relu10 = layers.ReLU()
        self.pool4 = layers.MaxPool2D(pool_size=2, strides=2, padding='SAME')
        self.drop4 = layers.Dropout(0.2)  # dropout层
        # -----------------------------------------------------------------------------------------
        self.conv11 = layers.Conv2D(filters=512, kernel_size=3, strides=1, padding='SAME', use_bias=False)
        self.bn11 = layers.BatchNormalization(momentum=0.9, epsilon=1e-5)
        self.relu11 = layers.ReLU()
        self.conv12 = layers.Conv2D(filters=512, kernel_size=3, strides=1, padding='SAME', use_bias=False)
        self.bn12 = layers.BatchNormalization(momentum=0.9, epsilon=1e-5)
        self.relu12 = layers.ReLU()
        self.conv13 = layers.Conv2D(filters=512, kernel_size=3, strides=1, padding='SAME', use_bias=False)
        self.bn13 = layers.BatchNormalization(momentum=0.9, epsilon=1e-5)
        self.relu13 = layers.ReLU()
        self.pool5 = layers.MaxPool2D(pool_size=2, strides=2, padding='SAME')
        self.drop5 = layers.Dropout(0.2)  # dropout层
        # -----------------------------------------------------------------------------------------
        self.flatten = layers.Flatten()
        self.dense1 = layers.Dense(units=4096, activation='relu')
        self.drop6 = layers.Dropout(0.2)  # dropout层
        self.dense2 = layers.Dense(units=4096,activation='relu')
        self.drop7 = layers.Dropout(0.2)  # dropout层
        self.dense3 = layers.Dense(units=class_num, activation='softmax')

    def call(self, inputs, training=False, **kwargs):
        # -----------------------------------------------------------------------------------------
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.bn2(x, training=training)
        x = self.relu2(x)
        x = self.pooll(x)
        x = self.drop1(x)

        # -----------------------------------------------------------------------------------------
        x = self.conv3(x)
        x = self.bn3(x, training=training)
        x = self.relu3(x)
        x = self.conv4(x)
        x = self.bn4(x, training=training)
        x = self.relu4(x)
        x = self.pool2(x)
        x = self.drop2(x)

        # -----------------------------------------------------------------------------------------
        x = self.conv5(x)
        x = self.bn5(x, training=training)
        x = self.relu5(x)
        x = self.conv6(x)
        x = self.bn6(x, training=training)
        x = self.relu6(x)
        x = self.conv7(x)
        x = self.bn7(x, training=training)
        x = self.relu7(x)
        x = self.pool3(x)
        x = self.drop3(x)

        # -----------------------------------------------------------------------------------------
        x = self.conv8(x)
        x = self.bn8(x, training=training)
        x = self.relu8(x)
        x = self.conv9(x)
        x = self.bn9(x, training=training)
        x = self.relu9(x)
        x = self.conv10(x)
        x = self.bn10(x, training=training)
        x = self.relu10(x)
        x = self.pool4(x)
        x = self.drop4(x)

        # -----------------------------------------------------------------------------------------
        x = self.conv11(x)
        x = self.bn11(x, training=training)
        x = self.relu11(x)
        x = self.conv12(x)
        x = self.bn12(x, training=training)
        x = self.relu12(x)
        x = self.conv13(x)
        x = self.bn13(x, training=training)
        x = self.relu13(x)
        x = self.pool5(x)
        x = self.drop5(x)

        # -----------------------------------------------------------------------------------------
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.drop6(x)
        x = self.dense2(x)
        x = self.drop7(x)
        x = self.dense3(x)

        return x


def vgg16(num_classes):
    return VGG16(num_classes)
