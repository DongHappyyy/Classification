# -*- coding:utf-8 _*-
"""
#  @Author: DongHao
#  @Date:   2020/11/26 15:58
#  @File:   train.py
"""
import os
import time
import glob
import tensorflow as tf
from tensorflow.keras import optimizers, losses, metrics
from GoogLeNet import GoogLeNet


def load_and_preprocess_image(img_path, label, height=224, width=224):
    label = tf.one_hot(label, depth=5)            # 采用softmax激活，label 为 one_hot
    img = tf.io.read_file(img_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, (height, width))
    img = (img - 0.5) / 0.5
    return img, label


@tf.function
def train_step(images, labels):
    with tf.GradientTape() as tape:
        aux1, aux2, output = model(images, training=True)
        loss1 = loss_func(labels, aux1)
        loss2 = loss_func(labels, aux2)
        loss3 = loss_func(labels, output)
        loss = loss1 * 0.3 + loss2 * 0.3 + loss3
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    train_loss.update_state(loss)
    train_metric.update_state(labels, output)


@tf.function
def validation_step(images, labels):
    _, _, predictions = model(images)
    batch_loss = loss_func(labels, predictions)
    validation_loss.update_state(batch_loss)
    validation_metric.update_state(labels, predictions)


print("\n============================================ Here is GoogLeNet ============================================\n")


# 定义用到的各种路径和变量
BATCH_SIZE = 32
cache = './cache.tf-data-train'
data_path = os.path.join(os.getcwd(), "../../")
model_save_path = os.path.join(data_path, "CheckPoint/GoogLeNet")
train_dir = os.path.join(data_path, "DataSet/flower_data/train")
validation_dir = os.path.join(data_path, "DataSet/flower_data/val")


print("\n============================================ 加载数据集 ==============================================\n")
# 获取分类类别
classes = [item for item in os.listdir(train_dir) if '.' not in item]
dic = dict((item, index) for index, item in enumerate(classes))

# 加载训练集，获取图片路径和标签, 使用并行化预处理num_parallel_calls 和预存数据prefetch来提升性能
train_images_path = list(glob.glob(train_dir+"/*/*.jpg"))
train_images_path = [str(path) for path in train_images_path]   # 转str
train_labels = [dic[path.split(os.sep)[-2]] for path in train_images_path]
total_train_images = len(train_images_path)
train_ds = tf.data.Dataset.from_tensor_slices((train_images_path, train_labels))
train_ds = train_ds.map(map_func=load_and_preprocess_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)\
    .cache().shuffle(buffer_size=total_train_images).repeat()\
    .batch(BATCH_SIZE).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

# 加载验证集，获取图片路径和标签, 使用并行化预处理num_parallel_calls 和预存数据prefetch来提升性能
validation_images_path = list(glob.glob(validation_dir+"/*/*.jpg"))
validation_images_path = [str(path) for path in validation_images_path]  # 转str
validation_labels = [dic[path.split(os.sep)[-2]] for path in validation_images_path]
total_validation_images = len(validation_images_path)
validation_ds = tf.data.Dataset.from_tensor_slices((validation_images_path, validation_labels))\
    .map(map_func=load_and_preprocess_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)\
    .cache().shuffle(buffer_size=total_validation_images)\
    .repeat().batch(BATCH_SIZE).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

print(train_ds)
print(validation_ds)
print("total_train_images is : ", total_train_images)
print("total_validation_images is : ", total_validation_images)


print("\n============================================== 创建模型 ==============================================\n")
# 定义模型
model = GoogLeNet(class_num=5)
model.build(input_shape=(None, 224, 224, 3))
model.summary()


print("\n============================================ 定义训练指标 ==============================================\n")
# 定义优化器和损失函数
optimizer = optimizers.Adam(learning_rate=0.01)
# loss_func = losses.SparseCategoricalCrossentropy()
loss_func = losses.CategoricalCrossentropy(from_logits=False)         # 采用softmax激活，label 为 one_hot

# 定义训练集评估指标
train_loss = metrics.Mean(name="train_loss")
# train_metric = metrics.SparseCategoricalAccuracy(name="train_accuracy")
train_metric = metrics.CategoricalAccuracy(name="train_accuracy")     # 采用softmax激活，label 为 one_hot

# 定义验证集评估指标
validation_loss = metrics.Mean(name='validation_loss')
# validation_metric = metrics.SparseCategoricalAccuracy(name="validation_accuracy")
validation_metric = metrics.CategoricalAccuracy(name="validation_accuracy")   # 采用softmax激活，label 为 one_hot


print("\n============================================== 开始训练 ==============================================\n")
epochs = 300
train_loss_results = []             # 保留结果用于绘制
train_accuracy_results = []
for epoch in range(epochs):
    train_loss.reset_states()
    train_metric.reset_states()
    validation_loss.reset_states()
    validation_metric.reset_states()

    total_A = total_train_images // BATCH_SIZE
    for index, (train_images, train_labels) in enumerate(train_ds):
        # print(train_images)     # shape=(32, 224, 224, 3)
        # print(train_labels)     # shape=(32, 5)
        train_step(train_images, train_labels)
        print("============= Training :   Total_Step = {},   Current_Step = {} =============\n".format(total_A, index))
        if index == total_A:
            break

    total_B = total_validation_images // BATCH_SIZE
    for index, (validation_images, validation_labels) in enumerate(validation_ds):
        validation_step(validation_images, validation_labels)
        print("============ Validation :   Total_Step = {},   Current_Step = {} ============\n".format(total_B, index))
        if index == total_B:
            break

    information = '\nEpoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}\n'
    print(information.format(epoch + 1, train_loss.result(), train_metric.result() * 100,
                             validation_loss.result(), validation_metric.result() * 100))

    train_loss_results.append(train_loss.result())
    train_accuracy_results.append(train_metric.result())

    if epoch % 30 == 0:
        temp_path = os.path.join(model_save_path, "googlenet_"+str(epoch+1))
        if not os.path.exists(temp_path):
            os.makedirs(temp_path)
        model.save(temp_path)

print("\n============================================ 训练结束，模型已保存 ============================================\n")
