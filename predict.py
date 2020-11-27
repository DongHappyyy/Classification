# -*- coding:utf-8 _*-
"""
#  @Author: DongHao
#  @Date:   2020/11/3 22:30
#  @File:   predict.py
"""
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from MobileNet.MobileNet import mobilenet
from ResNet.ResNet import resnet50
from VGG16.VGG16 import vgg16
import tensorflow as tf
from PIL import Image
import numpy as np
import json
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

im_height = 224
im_width = 224
modelname = "MobileNet"
classical_path = "./" + modelname + "/class_indices.json"
checkpoint_save_path = "./" + modelname + "/checkpoint/" + modelname + ".ckpt"
test_data_path = os.path.abspath(os.getcwd() + "/DataSet/testdata")

print("=============================这里是 " + modelname + " ===========================\n")
with open(classical_path, 'r') as json_file:
    class_indict = json.load(json_file)
    
print("=============================模型加载中===========================")
# model = resnet50(num_classes=5, include_top=True)
model = mobilenet(num_classes=5)
#model = vgg16(num_classes=5)
model.load_weights(checkpoint_save_path)
print("checkpoint path is ", checkpoint_save_path)
print("=============================模型加载完毕===========================\n")

print("=============================模型预测中===========================")
files = os.listdir(test_data_path)
for name in files:
    path = test_data_path + "\\" + str(name)
    img = Image.open(path)
    img = img.resize((im_width, im_height))
    img = np.array(img).astype(np.float32)
    img = (np.expand_dims(img, 0))
    result = model.predict(img)
    prediction = np.squeeze(result)
    predict_class = np.argmax(result)
    print(name)
    print(class_indict[str(predict_class)], prediction[predict_class])
    print("result is :", result)
print("=============================模型预测完毕===========================\n")
