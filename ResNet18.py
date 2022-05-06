
import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from tensorflow import keras
from tensorflow.python.keras.api._v2.keras import layers, optimizers, losses
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPool2D, Dropout, Flatten, Dense,ReLU,Add,GlobalAveragePooling2D,Input
from tensorflow.keras import Model
from tensorflow.keras.callbacks import EarlyStopping
from    tensorflow.keras import layers, optimizers,losses, datasets, Sequential
from PIL import Image

tf.random.set_seed(22)
np.random.seed(22)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
assert tf.__version__.startswith('2.')

# 导入一些具体的工具
from train import  load_train, normalize, denormalize
from valid import  load_valid, normalize,denormalize
from test import  load_test, normalize,denormalize
# from resnet import ResNet                   # 导入模型

# 预处理的函数，复制过来。
def preprocess(x,y):
    # x: 图片的路径，y：图片的数字编码
    x = tf.io.read_file(x)
    x = tf.image.decode_jpeg(x, channels=3) # RGBA
    x = tf.image.resize(x, [48, 48])

    # x = tf.image.random_flip_left_right(x)
    # x = tf.image.random_flip_up_down(x)
    # x = tf.image.random_crop(x, [224,224,3])

    # x: [0,255]=> -1~1
    x = tf.cast(x, dtype=tf.float32) / 255.
    x = normalize(x)
    y = tf.convert_to_tensor(y)
    y = tf.one_hot(y, depth=7)

    return x, y

batchsz = 32

# creat train db   一般训练的时候需要shuffle。其它是不需要的。
images, labels, table = load_train('train',mode='train')
db_train = tf.data.Dataset.from_tensor_slices((images, labels))  # 变成个Dataset对象。
db_train = db_train.shuffle(1000).map(preprocess).batch(batchsz) # map函数图片路径变为内容。
# crate validation db
images2, labels2, table = load_valid('valid',mode='val')
db_val = tf.data.Dataset.from_tensor_slices((images2, labels2))
db_val = db_val.map(preprocess).batch(batchsz)
# create test db
images3, labels3, table = load_test('test',mode='test')
db_test = tf.data.Dataset.from_tensor_slices((images3, labels3))
db_test = db_test.map(preprocess).batch(batchsz)





# def conv_batchnorm_relu(x, filters, kernel_size, strides):
#     x = Conv2D(filters=filters,
#                kernel_size=kernel_size,
#                strides=strides,
#                padding='same')(x)
#     x = BatchNormalization()(x)
#     x = ReLU()(x)
#     return x
#
#
# def identity_block(tensor, filters):
#     x = conv_batchnorm_relu(tensor, filters=filters, kernel_size=3, strides=1)
#     x = conv_batchnorm_relu(x, filters=filters, kernel_size=3, strides=1)
#
#     x = Add()([x, tensor])
#     x = ReLU()(x)
#     return x
#
#
# def projection_block(tensor, filters, strides):
#     # left stream
#     x = conv_batchnorm_relu(tensor, filters=filters, kernel_size=3, strides=1)
#     x = conv_batchnorm_relu(x, filters=filters, kernel_size=3, strides=strides)
#
#     # right stream
#     shortcut = Conv2D(filters, kernel_size=1, strides=strides, padding='same')(tensor)
#     shortcut = BatchNormalization()(shortcut)
#
#     x = Add()([x, shortcut])
#     x = ReLU()(x)
#     return x
#
#
# def resnet_block(x, filters, reps, strides):
#     x = projection_block(x, filters=filters, strides=strides)
#     for _ in range(reps - 1):  # the -1 is because the first block was a Conv one
#         x = identity_block(x, filters=filters)
#     return x
#
#
# input = Input(shape=(48, 48, 3))
#
# x = conv_batchnorm_relu(input, filters=32, kernel_size=3, strides=2)  # 7x7, 32, strides 2
# # x = MaxPool2D(pool_size=3, strides=2, padding='same')(x)  # 3x3 max mool, strides 2
#
# x = resnet_block(x, filters=32, reps=2, strides=2)
# x = resnet_block(x, filters=64, reps=2, strides=2)  # s=2 ([2]: conv3_1)
# x = resnet_block(x, filters=96, reps=2, strides=2)  # s=2 ([2]: conv4_1)
# x = resnet_block(x, filters=128, reps=2, strides=2)  # s=2 ([2]: conv5_1)
#
# x = GlobalAveragePooling2D()(x)
#
# pred = Dense(7, activation='softmax')(x)
#
# from tensorflow.keras import Model
#
# model = Model(inputs=input, outputs=pred)

model = tf.keras.Sequential ([
    layers.Conv2D(96, (5, 5),strides=2, input_shape=(48, 48, 3)),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.MaxPool2D(pool_size=3,strides=2,padding='same'),

    layers.Conv2D(128, (3, 3)),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.MaxPool2D(pool_size=3,strides=2,padding='same'),

    layers.Conv2D(192, (3, 3),padding='same'),
    layers.BatchNormalization(),
    layers.Activation('relu'),

    layers.Conv2D(192, (3, 3),padding='same'),
    layers.BatchNormalization(),
    layers.Activation('relu'),


    layers.Conv2D(128, (3, 3),padding='same'),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    # layers.MaxPool2D(pool_size=3,strides=2),


    layers.GlobalAveragePooling2D(),

    # layers.Flatten(),
    # layers.Dense(4096, activation='relu'),
    # layers.Dropout(0.5),
    # layers.Dense(4096, activation='relu'),
    # layers.Dropout(0.5),
    layers.Dense(7,activation='softmax')
])




model.summary()

# 网络的装配。
exponential_decay = tf.keras.optimizers.schedules.ExponentialDecay(
                        initial_learning_rate=0.001, decay_steps=478, decay_rate=0.90)

model.compile(optimizer=optimizers.Adam(exponential_decay),
               loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
               metrics=['acc'])

# 完成标准的train，val, test;
# 标准的逻辑必须通过db_val挑选模型的参数，就需要提供一个earlystopping技术，
history = model.fit(db_train, validation_data=db_val, validation_freq=1, epochs=80,
                    )   # 1个epoch验证1次。触发了这个事情，提前停止了。
model.evaluate(db_test)
model.save('ResNet18.h5')

# 显示训练集和验证集的acc和loss曲线
# acc = history.history['accuracy']
# val_acc = history.history['val_accuracy']
# loss = history.history['loss']
# val_loss = history.history['val_loss']

plt.subplot(1, 2, 1)
plt.plot(history.epoch, history.history.get('acc'),label='acc')
plt.plot(history.epoch, history.history.get('val_acc'),label='val_acc')
plt.title('Training and Validation Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.epoch, history.history.get('loss'),label='loss')
plt.plot(history.epoch, history.history.get('val_loss'),label='val_loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()