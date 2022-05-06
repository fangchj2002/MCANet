from tensorflow.keras.layers import GlobalAveragePooling2D, GlobalMaxPooling2D, Reshape, Dense, multiply, Permute, Concatenate, \
    Conv2D, Add, Activation, Lambda,Conv1D
from tensorflow.keras import backend as K
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
import math
def attach_attention_module(net, attention_module):
    if attention_module == 'se_block':  # SE_block
        net = se_block(net)
    elif attention_module == 'cbam_block':  # CBAM_block
        net = cbam_block(net)
    else:
        raise Exception("'{}' is not supported attention module!".format(attention_module))

    return net


def se_block(input_feature, ratio=8):
    """Contains the implementation of Squeeze-and-Excitation(SE) block.
    As described in https://arxiv.org/abs/1709.01507.
    """

    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    channel = input_feature.shape[channel_axis]

    se_feature = GlobalAveragePooling2D()(input_feature)
    se_feature = Reshape((1, 1, channel))(se_feature)
    assert se_feature.shape[1:] == (1, 1, channel)
    se_feature = Dense(channel // ratio,
                       activation='relu',
                       kernel_initializer='he_normal',
                       use_bias=True,
                       bias_initializer='zeros')(se_feature)
    assert se_feature.shape[1:] == (1, 1, channel // ratio)
    se_feature = Dense(channel,
                       activation='sigmoid',
                       kernel_initializer='he_normal',
                       use_bias=True,
                       bias_initializer='zeros')(se_feature)
    assert se_feature.shape[1:] == (1, 1, channel)
    if K.image_data_format() == 'channels_first':
        se_feature = Permute((3, 1, 2))(se_feature)

    se_feature = multiply([input_feature, se_feature])
    return se_feature

def feature_block(input_feature):
    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    channel = input_feature.shape[channel_axis]

    s_feature = GlobalAveragePooling2D()(input_feature)
    a = Reshape((1, channel))(s_feature)
    x=Conv1D(channel,1,padding="same",activation='sigmoid')(a)
    x=Reshape((1, 1,channel))(x)
    shared_layer_two = Conv2D(channel, 1, strides=1, use_bias=False, activation="sigmoid")(x)


 #   h_feature = Reshape((1, 1,length))(e_feature)
 #   shared_layer_one = Conv2D(fbank // ratio, 3, strides=1, use_bias=False, activation="relu")(w_feature.shape[1:])
 #   shared_layer_two=Conv1D(fbank,3, strides=1,use_bias=False)(shared_layer_one)
    #   GAP_out=shared_layer_two
    #   shared_layer_one = Conv1D(fbank // ratio, 3, strides=1, use_bias=False, activation="relu")(x1)
    #   shared_layer_three = Conv1D(fbank, 3, strides=1, use_bias=False)(shared_layer_one)
    #   GMP_out =shared_layer_three
 #   assert h_feature.shape[1:] == (1, 1,length)
 #   c,h,w = w_feature.shape[1:]
  #  c1,w1,h1 = h_feature.shape[1:]
  #  b = [tf.squeeze(t, axis=1) for t in tf.split(h_feature, num_or_size_splits=1, axis=1)]
 #   b = tf.concat(b, axis=0)
    h_feature = Activation('sigmoid')(shared_layer_two)
    if K.image_data_format() == "channels_first":
       h_feature = Permute((3, 1, 2))(h_feature)
    h=multiply([input_feature, h_feature])
   # w_feature = Activation('sigmoid')(GMP_out)
   # w=multiply([h, w_feature])
  #  result = channel_attention(input_feature, ratio=8)
   # se=se_block(input_feature, ratio=8)
    # if K.image_data_format() == "channels_first":
    #      wh_feature = Permute((3, 1, 2))(wh_feature)
    return h

def time_block(input_feature):
    channel_axis = 1
    channel = input_feature.shape[channel_axis]

    s_feature = GlobalAveragePooling2D()(input_feature)
    a = Reshape((1, channel))(s_feature)
    x=Conv1D(channel,1,padding="same",activation='sigmoid')(a)
    x=Reshape((1, 1,channel))(x)
    shared_layer_two = Conv2D(channel, 1, strides=1, use_bias=False, activation="sigmoid")(x)


 #   h_feature = Reshape((1, 1,length))(e_feature)
 #   shared_layer_one = Conv2D(fbank // ratio, 3, strides=1, use_bias=False, activation="relu")(w_feature.shape[1:])
 #   shared_layer_two=Conv1D(fbank,3, strides=1,use_bias=False)(shared_layer_one)
    #   GAP_out=shared_layer_two
    #   shared_layer_one = Conv1D(fbank // ratio, 3, strides=1, use_bias=False, activation="relu")(x1)
    #   shared_layer_three = Conv1D(fbank, 3, strides=1, use_bias=False)(shared_layer_one)
    #   GMP_out =shared_layer_three
 #   assert h_feature.shape[1:] == (1, 1,length)
 #   c,h,w = w_feature.shape[1:]
  #  c1,w1,h1 = h_feature.shape[1:]
  #  b = [tf.squeeze(t, axis=1) for t in tf.split(h_feature, num_or_size_splits=1, axis=1)]
 #   b = tf.concat(b, axis=0)
    h_feature = Activation('sigmoid')(shared_layer_two)
    if K.image_data_format() == "channels_first":
       h_feature = Permute((3, 1, 2))(h_feature)
    h=multiply([input_feature, h_feature])
   # w_feature = Activation('sigmoid')(GMP_out)
   # w=multiply([h, w_feature])
  #  result = channel_attention(input_feature, ratio=8)
   # se=se_block(input_feature, ratio=8)
    # if K.image_data_format() == "channels_first":
    #      wh_feature = Permute((3, 1, 2))(wh_feature)
    return h


def frame_block(input_feature, ratio=8):
    """Contains the implementation of Convolutional Block Attention Module(CBAM) block.
    As described in https://arxiv.org/abs/1807.06521.
    """

    frame_feature = se_block(input_feature, ratio)
    frame_feature = feature_block(input_feature,ratio)
    return frame_feature

def cbam_block(input_feature, ratio=8):
    """Contains the implementation of Convolutional Block Attention Module(CBAM) block.
    As described in https://arxiv.org/abs/1807.06521.
    """

    cbam_feature = channel_attention(input_feature, ratio)
    cbam_feature = spatial_attention(input_feature)
    return cbam_feature


def channel_attention(input_feature, ratio=8):
    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    channel = input_feature.shape[channel_axis]

    shared_layer_one = Dense(channel // ratio,
                             activation='relu',
                             kernel_initializer='he_normal',
                             use_bias=True,
                             bias_initializer='zeros')
    shared_layer_two = Dense(channel,
                             kernel_initializer='he_normal',
                             use_bias=True,
                             bias_initializer='zeros')

    avg_pool = GlobalAveragePooling2D()(input_feature)
    avg_pool = Reshape((1, 1, channel))(avg_pool)
    assert avg_pool.shape[1:] == (1, 1, channel)
    avg_pool = shared_layer_one(avg_pool)
    assert avg_pool.shape[1:] == (1, 1, channel // ratio)
    avg_pool = shared_layer_two(avg_pool)
    assert avg_pool.shape[1:] == (1, 1, channel)

    max_pool = GlobalMaxPooling2D()(input_feature)
    max_pool = Reshape((1, 1, channel))(max_pool)
    assert max_pool.shape[1:] == (1, 1, channel)
    max_pool = shared_layer_one(max_pool)
    assert max_pool.shape[1:] == (1, 1, channel // ratio)
    max_pool = shared_layer_two(max_pool)
    assert max_pool.shape[1:] == (1, 1, channel)

    cbam_feature = Add()([avg_pool, max_pool])
    cbam_feature = Activation('sigmoid')(cbam_feature)

    if K.image_data_format() == "channels_first":
        cbam_feature = Permute((3, 1, 2))(cbam_feature)
    result=multiply([input_feature, cbam_feature])
    return result


def spatial_attention(input_feature):
    result=feature_block(input_feature)
    kernel_size = 7

    if K.image_data_format() == "channels_first":
        channel = input_feature.shape[1]
        cbam_feature = Permute((2, 3, 1))(input_feature)
    else:
        channel = input_feature.shape[-1]
        cbam_feature = input_feature

    avg_pool = Lambda(lambda x: K.mean(x, axis=3, keepdims=True))(cbam_feature)
    assert avg_pool.shape[-1] == 1
    max_pool = Lambda(lambda x: K.max(x, axis=3, keepdims=True))(cbam_feature)
    assert max_pool.shape[-1] == 1
    concat = Concatenate(axis=3)([avg_pool, max_pool])
    assert concat.shape[-1] == 2
    cbam_feature = Conv2D(filters=1,
                          kernel_size=kernel_size,
                          strides=1,
                          padding='same',
                          activation='sigmoid',
                          kernel_initializer='he_normal',
                          use_bias=False)(concat)
    assert cbam_feature.shape[-1] == 1

    if K.image_data_format() == "channels_first":
        cbam_feature = Permute((3, 1, 2))(cbam_feature)
    b=multiply([cbam_feature, result])
    return b

