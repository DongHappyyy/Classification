# -*- coding:utf-8 _*-
"""
#  @Author: DongHao
#  @Date:   2020/11/3 15:29
#  @File:   train.py
"""
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from model import resnet50
import tensorflow as tf
from PIL import Image
import numpy as np
import json
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

im_height = 224
im_width = 224
classical_path = "./class_indices.json"
checkpoint_save_path = "./checkpoint/ResNet.ckpt"
test_data_path = os.path.abspath(os.getcwd() + "/../DataSet/testdata")
train_dir = os.path.abspath(os.getcwd() + "/../DataSet/flower_data/train")
validation_dir = os.path.abspath(os.getcwd() + "/../DataSet/flower_data/val")

print("\n=============================这里是 ResNet ===========================\n")

train_image_generator = ImageDataGenerator(horizontal_flip=True)
train_data_gen = train_image_generator.flow_from_directory(directory=train_dir, batch_size=3306, shuffle=True,
                                                           target_size=(im_height, im_width), class_mode='categorical')

class_indices = train_data_gen.class_indices    # get class dict
inverse_dict = dict((val, key) for key, val in class_indices.items())
json_str = json.dumps(inverse_dict, indent=4)
with open(classical_path, 'w') as json_file:
    json_file.write(json_str)

validation_image_generator = ImageDataGenerator()
val_data_gen = validation_image_generator.flow_from_directory(directory=validation_dir, batch_size=364, shuffle=False,
                                                              target_size=(im_height, im_width), class_mode='categorical')

print("class_indices : ", class_indices)
print("total_train : ", train_data_gen.n)
print("total_val : ", val_data_gen.n)

print("===============================数据加载中===========================")
train_x, train_y = next(train_data_gen)
validation_x, validation_y = next(val_data_gen)
print("=============================数据加载完毕===========================\n")

print("=============================模型加载中===========================")
model = resnet50(num_classes=5, include_top=True)
model.build((None, 224, 224, 3))
model.summary()
# model.build(input_shape) # `input_shape` is the shape of the input data, e.g. input_shape = (None, 32, 32, 3)
model.compile(
    optimizer=tf.keras.optimizers.SGD(learning_rate=0.0001),
    loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
    metrics=["accuracy"])
print("=============================模型加载完毕===========================\n")

print("=============================模型训练中===========================")
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_save_path, save_weights_only=True, save_best_only=True)
model.fit(train_x, train_y, batch_size=32, epochs=1000, validation_data=(validation_x, validation_y), callbacks=[cp_callback])
model.summary()
print("=============================模型训练完毕===========================\n")
