import logging
import tensorflow.keras
import tensorflow.keras.backend as K
import math
from tensorflow.keras import layers,Sequential
from tensorflow.keras import regularizers,models
from tensorflow.keras.layers import Input, GRU,LSTM,Dropout,multiply,Flatten,concatenate,GlobalAveragePooling1D,MaxPool2D,Activation,AveragePooling2D,Add,ZeroPadding2D,Bidirectional
from tensorflow.keras.layers import Conv2D,GlobalAveragePooling2D,MaxPooling2D,Concatenate,LeakyReLU,Convolution2D,Permute
from tensorflow.keras.layers import Lambda, Dense, RepeatVector
from tensorflow.keras.layers import Reshape,Layer
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.models import Model
from constants import *
import  tensorflow as tf
from attention_module import cbam_block,se_block,feature_block,frame_block,spatial_attention,time_block

def clipped_relu(inputs):
    return Lambda(lambda y: K.minimum(K.maximum(y, 0), 20))(inputs)

def identity_block(input_tensor, kernel_size, filters, stage, block):
    conv_name_base = 'res{}_{}_branch'.format(stage, block)

    x = Conv2D(filters,
                   kernel_size=kernel_size,
                   strides=1,
                   activation=None,
                   padding='same',
                   kernel_initializer='glorot_uniform',
                   kernel_regularizer=regularizers.l2(l=0.00001),
                   name=conv_name_base + '_2a')(input_tensor)
    x = BatchNormalization(name=conv_name_base + '_2a_bn')(x)
    x = clipped_relu(x)

    x = Conv2D(filters,
                   kernel_size=kernel_size,
                   strides=1,
                   activation=None,
                   padding='same',
                   kernel_initializer='glorot_uniform',
                   kernel_regularizer=regularizers.l2(l=0.00001),
                   name=conv_name_base + '_2b')(x)
    x = BatchNormalization(name=conv_name_base + '_2b_bn')(x)


    x = layers.add([x, input_tensor])
    x = clipped_relu(x)
    return x

def convolutional_model(input_shape=(NUM_FRAMES,64, 1),    #input_shape(32,32,3)
                        batch_size=BATCH_SIZE * TRIPLET_PER_BATCH , num_frames=NUM_FRAMES):
    # http://cs231n.github.io/convolutional-networks/
    # conv weights
    # #params = ks * ks * nb_filters * num_channels_input

    # Conv128-s
    # 5*5*128*128/2+128
    # ks*ks*nb_filters*channels/strides+bias(=nb_filters)

    # take 100 ms -> 4 frames.
    # if signal is 3 seconds, then take 100ms per 100ms and average out this network.
    # 8*8 = 64 features.

    # used to share all the layers across the inputs

    # num_frames = K.shape() - do it dynamically after.

    def conv_and_res_block(inp, filters, stage):
        conv_name = 'conv{}-s'.format(filters)
        o = Conv2D(filters,
                       kernel_size=5,
                       strides=2,
                       padding='same',
                       kernel_initializer='glorot_uniform',
                       kernel_regularizer=regularizers.l2(l=0.00001), name=conv_name)(inp)
        o = BatchNormalization(name=conv_name + '_bn')(o)
        o = clipped_relu(o)
        for i in range(3):
            o = identity_block(o, kernel_size=3, filters=filters, stage=stage, block=i)
        return o

    def cnn_component(inp):
        x_ = conv_and_res_block(inp, 64, stage=1)
        x_ = conv_and_res_block(x_, 128, stage=2)
        x_ = conv_and_res_block(x_, 256, stage=3)
        x_ = conv_and_res_block(x_, 512, stage=4)
        return x_

    inputs = Input(shape=input_shape)  # TODO the network should be definable without explicit batch shape
    #x = Lambda(lambda y: K.reshape(y, (batch_size*num_frames,input_shape[1], input_shape[2], input_shape[3])), name='pre_reshape')(inputs)
    x = cnn_component(inputs)  # .shape = (BATCH_SIZE , num_frames/16, 64/16, 512)
    #x = Reshape((-1,2048))(x)
    x = Lambda(lambda y: K.reshape(y, (-1, math.ceil(num_frames/16), 2048)), name='reshape')(x)
    x = Lambda(lambda y: K.mean(y, axis=1), name='average')(x)  #shape = (BATCH_SIZE, 512)
    x = Dense(512, name='affine')(x)  # .shape = (BATCH_SIZE , 512)
    x = Lambda(lambda y: K.l2_normalize(y, axis=1), name='ln')(x)

    model = Model(inputs, x, name='convolutional')

    print(model.summary())
    return model

def basic_block(filters, kernel_size=3, is_first_block=True):
    stride = 1
    if is_first_block:
        stride = 2

    def f(x):
        # f(x) named y
        # 1st Conv
        y = ZeroPadding2D(padding=1)(x)
        y = Conv2D(filters, kernel_size, strides=stride, kernel_initializer='he_normal')(y)
        y = BatchNormalization()(y)
        y = Activation("relu")(y)
        # 2nd Conv
        y = ZeroPadding2D(padding=1)(y)
        y = Conv2D(filters, kernel_size, kernel_initializer='he_normal')(y)
        y = BatchNormalization()(y)

        # f(x) + x
        if is_first_block:
            shortcut = Conv2D(filters, kernel_size=1, strides=stride, kernel_initializer='he_normal')(x)
            shortcut = BatchNormalization()(shortcut)
        else:
            shortcut = x

        y = Add()([y, shortcut])
        y = Activation("relu")(y)

        return y

    return f

def ResNet18(input_shape=(NUM_FRAMES,64, 1), batch_size=BATCH_SIZE * TRIPLET_PER_BATCH,num_frames=NUM_FRAMES):
    input_layer = Input(shape=input_shape, name="input")

    # Conv1
    x = layers.ZeroPadding2D(padding=(3, 3), name='conv1_pad')(input_layer)
    x = layers.Conv2D(64, (7, 7),
                      strides=(2, 2),
                      padding='valid',
                      kernel_initializer='he_normal',
                      name='conv1')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.ZeroPadding2D(padding=(1, 1), name='pool1_pad')(x)
    x = layers.MaxPooling2D((3, 3), strides=(2, 2))(x)

    # Conv2
    x = basic_block(filters=64)(x)
    x = basic_block(filters=64, is_first_block=False)(x)

    # Conv3
    x = basic_block(filters=128)(x)
    x = basic_block(filters=128, is_first_block=False)(x)

    # Conv4
    x = basic_block(filters=256)(x)
    x = basic_block(filters=256, is_first_block=False)(x)

    # Conv5
    x = basic_block(filters=512)(x)
    x = basic_block(filters=512, is_first_block=False)(x)

    x = GlobalAveragePooling2D(name="feature")(x)
  #  output_layer = Dense(num_classes, activation='softmax')(x)

    model = Model(input_layer,x,name='resnet18')
    print(model.summary())
    return model


def recurrent_model(input_shape=(NUM_FRAMES, 64, 1),
                    batch_size=BATCH_SIZE * TRIPLET_PER_BATCH ,num_frames=NUM_FRAMES):
    inputs = Input(shape=input_shape)
    #x = Permute((2,1))(inputs)
    x = Conv2D(64,kernel_size=5,strides=2,padding='same',kernel_initializer='glorot_uniform',kernel_regularizer=regularizers.l2(l=0.0001))(inputs)
    x = BatchNormalization()(x)  #shape = (BATCH_SIZE , num_frames/2, 64/2, 64)
    x = clipped_relu(x)
    x = Lambda(lambda y: K.reshape(y, (-1, math.ceil(num_frames / 2), 2048)), name='reshape')(x) #shape = (BATCH_SIZE , num_frames/2, 2048)
    x = GRU(1024,return_sequences=True)(x)  #shape = (BATCH_SIZE , num_frames/2, 1024)
    x = GRU(1024,return_sequences=True)(x)
    x = GRU(1024,return_sequences=True)(x)  #shape = (BATCH_SIZE , num_frames/2, 1024)
  #  x = SeqSelfAttention(attention_activation='softmax')
    x = Lambda(lambda y: K.mean(y, axis=1), name='average')(x) #shape = (BATCH_SIZE, 1024)
    x = Dense(512)(x)  #shape = (BATCH_SIZE, 512)
    x = Lambda(lambda y: K.l2_normalize(y, axis=1), name='ln')(x)

    model = Model(inputs,x,name='recurrent')

    print(model.summary())
    return model

def Lenet(input_shape=(NUM_FRAMES, 64, 1),
                    batch_size=BATCH_SIZE * TRIPLET_PER_BATCH ,num_frames=NUM_FRAMES):
    inputs = Input(shape=input_shape)
    x=Conv2D(6, (5, 5))(inputs)
    x=Activation('tanh')(x)
    x=MaxPool2D(pool_size=2, strides=2)(x)


    x=Conv2D(16, (5, 5))(x)
    x = Activation('tanh')(x)
    x = MaxPool2D(pool_size=2, strides=2)(x)

    x = Flatten()(x)
    x = Dense(120, activation='tanh')(x)
    x = Dense(84, activation='tanh')(x)
   # x = TimeDistributed(Flatten(), name='timedistrib')(x)
   # x = GRU(512, return_sequences=True)(x)
   # x = Lambda(lambda y: K.mean(y, axis=1), name='average')(x)
   # x= GlobalAveragePooling1D()(x)
  #  x = Lambda(lambda y: K.l2_normalize(y, axis=1), name='ln')(x)
    #x=Flatten()(x)
   # x=Dropout(0.5)(x)
    #x=Dense(4096, activation='relu')(x)
  #  x=Dropout(0.5)(x)

    model = Model(inputs, x, name='Lenet')
    print(model.summary())
    return model

def Alexnet(input_shape=(NUM_FRAMES, 64, 1),
                    batch_size=BATCH_SIZE * TRIPLET_PER_BATCH ,num_frames=NUM_FRAMES):
    inputs = Input(shape=input_shape)
    x = Conv2D(96, (11, 11), strides=4)(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPool2D(pool_size=3, strides=2)(x)

    x = Conv2D(256, (5, 5), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
   # x = MaxPool2D(pool_size=3, strides=2)(x)

    x = Conv2D(384, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)


    x = Conv2D(384, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(256, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
  #  x = MaxPool2D(pool_size=3, strides=2)(x)

    x = Flatten()(x)
    x = Dense(4096, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(4096, activation='relu')(x)
    x = Dropout(0.5)(x)
    model = Model(inputs, x, name='Alexnet')
    print(model.summary())
    return model


def regurlarized_padded_conv(*args, **kwargs):
    return layers.Conv2D(*args, **kwargs, padding="same",
                         use_bias=False,
                         kernel_initializer="he_normal",
                         kernel_regularizer=regularizers.l2(1e-4))

def lrn(input, radius=2, alpha=.001, beta=.75, name='LRN', bias=1.0):
    return tf.nn.local_response_normalization(input,depth_radius=radius,alpha=alpha,beta=beta,bias=bias,name=name)

def VGG_M():
    model = Sequential(
        [Conv2D(96, (7, 7), strides=2, input_shape=(NUM_FRAMES, 64, 1)),
         Activation('relu'),
         # LRN
         Lambda(lrn),
         MaxPooling2D(pool_size=3, strides=2),

         Conv2D(256, (5, 5), strides=2,activation="relu"),
         ZeroPadding2D((1,1)),
         # LRN
         Lambda(lrn),
         MaxPooling2D(pool_size=3,strides=2),

         Conv2D(512, (3, 3), activation='relu',padding="same"),
         Conv2D(512, (3, 3), activation='relu',padding="same"),
         Conv2D(512, (3, 3), activation='relu',padding="same"),
         MaxPooling2D(pool_size=3,strides=2),
         Flatten(),
         Dense(4096, activation='relu'),
         Dropout(.5),
         Dense(4096, activation='relu'),
         Dropout(.5),
         Dense(240, activation='softmax')
         ]
    )
    return model
def dvector(input_shape=(NUM_FRAMES, 64,1),batch_size=BATCH_SIZE * TRIPLET_PER_BATCH ,num_frames=NUM_FRAMES):
    inputs = Input(shape=input_shape)
    x=Flatten()(inputs)
   # x = Lambda(lambda y: K.reshape(y, (-1, math.ceil(10240),1)), name='Reshape')(inputs)
    x=Dense(256, name="dense1",activation="relu")(x)


    x=Dense(256, name="dense2",activation="relu")(x)
    x=Dropout(0.5)(x)

    x=Dense(256, name="dense3",activation="relu")(x)
    x = Dropout(0.5)(x)
  #  x = GlobalAveragePooling1D()(x)
    x1=Dense(1211, name="dense4")(x)
    x=Activation('softmax', name="activation3")(x1)
    model=Model(inputs, x,name="D-vector")

    print(model.summary())
    return model


def inception_block(x, filters):

    t1 = Conv2D(filters=filters[0], kernel_size=3,dilation_rate=(1,1),strides=1,activation="relu", padding='same')(x)
    # t1 = BatchNormalization()(t1)
  # t1 = LeakyReLU(0.2)(t1)

    t2 = Conv2D(filters=filters[1], kernel_size=3,dilation_rate=(2,2),strides=1,activation="relu",padding='same')(x)
    # t2 = BatchNormalization()(t2)
    #t2 = Activation('relu')(t2)
  #  t2 = LeakyReLU(0.2)(t2)

    t3 = Conv2D(filters=filters[2], kernel_size=3,dilation_rate=(3,3),strides=1,activation="relu",padding='same')(x)
   # t3 = LeakyReLU(0.2)(t3)

    t4 = Conv2D(filters=filters[3], kernel_size=1, strides=1, activation="relu",padding='same')(x)
   # t4 = LeakyReLU(0.2)(t4)
    output = Concatenate()([t1, t2, t3, t4])
    return output

def fAlexnet(input_shape=(NUM_FRAMES, 64,1),
                    batch_size=BATCH_SIZE * TRIPLET_PER_BATCH ,num_frames=NUM_FRAMES):
    inputs = Input(shape=input_shape)
    # x_shortcut = inputs
    x = regurlarized_padded_conv(64, kernel_size=5,strides=1)(inputs)
  #  x = Conv2D(64, (5, 5), strides=1, padding='same')(inputs)
    x = BatchNormalization()(x)
    x1 = Activation('relu')(x)
   # x1=LeakyReLU(0.2)(x)
 #   x = cbam_block(x1, ratio=8)
    x2 =feature_block(x1)
    x = inception_block(x2,filters=[32,32, 32, 32])
  #  x = MaxPool2D(pool_size=3, strides=2)(x)
   # x = inception_block(x1, filters=[32, 32, 32, 32])
   # x = feature_block(x)
    x = Conv2D(64, (3, 3), strides=1, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    #x = Conv2D(128, (3, 3), padding='same')(x)
   # x = BatchNormalization()(x)
   # x = Activation('relu')(x)
   # x = MaxPool2D(pool_size=3, strides=2)(x)

   # x_shortcut=Conv2D(128, (1, 1),strides=2, padding='valid')(x_shortcut)
  #  x_shortcut = BatchNormalization()(x_shortcut)
    x = layers.add([x, x1])
    x = Activation('relu')(x)
    x = Conv2D(128, (3, 3), strides=1,padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
  #  x = Conv2D(128, (3, 3),padding='same')(x)
  #  x = Activation('relu')(x)
   # x = layers.add([x2, x])
    x = Lambda(lambda y: K.reshape(y, (-1, math.ceil(160),8192)), name='Reshape')(x)
   # x = GRU(1024, return_sequences=True)(x)
   # x = GRU(1024, return_sequences=True)(x)
  #   x = GRU(1024, return_sequences=True)(x)
    x = GlobalAveragePooling1D()(x)
    x = Lambda(lambda y: K.l2_normalize(y, axis=1), name='ln')(x)
    model = Model(inputs,x, name='MCN')
    print(model.summary())
    return model

def TCM(input_shape=(NUM_FRAMES, 64,1),
                    batch_size=BATCH_SIZE * TRIPLET_PER_BATCH ,num_frames=NUM_FRAMES,num_channels=64):
    inputs = Input(shape=input_shape)
    x = regurlarized_padded_conv(num_channels, kernel_size=5, strides=1)(inputs)
    x = BatchNormalization()(x)
    x1 = Activation('relu')(x)
    subset_x = []
    w=num_channels//4
    for i in range(4):
        slice_x = Lambda(lambda x: x[..., i * w:(i + 1) * w])(x1)
        if i == 1 :
            slice_x = layers.add([slice_x, subset_x[-1]])
            slice_x = feature_block(slice_x)
        elif i == 2:
            slice_x = layers.add([slice_x, subset_x[-1]])
            slice_x = layers.add([slice_x, subset_x[1]])
            slice_x = feature_block(slice_x)
        elif i == 3:
            slice_x = layers.add([slice_x, subset_x[-1]])
            slice_x = layers.add([slice_x, subset_x[1]])
            slice_x = layers.add([slice_x, subset_x[2]])
            slice_x = feature_block(slice_x)
        subset_x.append(slice_x)
        # 将subset_x中保存的 y1，y2, y3, y4 进行 concat
    x = Concatenate()(subset_x)
    x = inception_block(x, filters=[32, 32, 32, 32])
    x = Conv2D(64, (3, 3), strides=1, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = layers.add([x, x1])
    x = Activation('relu')(x)
    x = Conv2D(128, (3, 3), strides=1, padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
#    x = layers.add([x, x1])
    x = Lambda(lambda y: K.reshape(y, (-1, math.ceil(160), 8192)), name='Reshape')(x)
   # x = Bidirectional(LSTM(300, return_sequences=True))(x)
    x = GlobalAveragePooling1D()(x)
    x = Lambda(lambda y: K.l2_normalize(y, axis=1), name='ln')(x)
    model = Model(inputs, x, name='TCM')
    print(model.summary())
    return model

def Res2Net(input_shape=(NUM_FRAMES, 64,1),
                    batch_size=BATCH_SIZE * TRIPLET_PER_BATCH ,num_frames=NUM_FRAMES,num_channels=64):
    inputs = Input(shape=input_shape)
    x = Conv2D(64, (1, 1), strides=1, padding='same')(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    subset_x = []
    n = 64
    w = n // 4
    for i in range(4):
        slice_x = Lambda(lambda x: x[..., i * w:(i + 1) * w])(x)
        if i > 1:
            slice_x = Add()([slice_x, subset_x[-1]])
        if i > 0:
            slice_x = BatchNormalization()(slice_x)
            slice_x = Activation('relu')(slice_x)
            slice_x = Conv2D(w, 3, kernel_initializer='he_normal', padding='same', use_bias=False)(slice_x)
        subset_x.append(slice_x)
    x = Concatenate()(subset_x)
    x = Conv2D(128, (1, 1), strides=1, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
 #   x = se_block(x)
    x = Lambda(lambda y: K.reshape(y, (-1, math.ceil(160), 8192)), name='Reshape')(x)
    x = GlobalAveragePooling1D()(x)
    x = Lambda(lambda y: K.l2_normalize(y, axis=1), name='ln')(x)
    model = Model(inputs, x, name='Res2net-se')
    print(model.summary())
    return model

if __name__ == '__main__':
   TCM()






