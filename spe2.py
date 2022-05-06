from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, MaxoutDense, GRU
from keras.models import load_model
from keras.layers.normalization import BatchNormalization
from keras import Model
from keras.layers.core import Reshape,Masking,Lambda,Permute
from keras.layers import Input,Dense,Flatten
import keras
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers.wrappers import TimeDistributed
from keras.optimizers import SGD, Adam
from keras.callbacks import ModelCheckpoint, TensorBoard
import random

import keras.backend as K
import numpy as np
import librosa
import python_speech_features as psf
import os
import tensorflow as tf


filePath = "E:/train-clean-100"
# filePath = "data/test"
nClass = len(os.listdir(filePath))


def getwavPathAndwavLabel(filePath):
    wavPath = []
    wavLabel = []
    testPath = []
    testLable = []
    files = os.listdir(filePath)
    lab = 0
    for file in files:
        wav = os.listdir(filePath+"/"+file+"/")
        for j in range(len(wav)):
            fileType = wav[j].split("/")[-1].split('.')[0]
            print(fileType)
            if float(j)>0.01*len(wav):
                if fileType=="wav":
                    wavLabel.append(lab)
                    wavPath.append(filePath+"/"+file+"/"+wav[j])
            else:
                testLable.append(lab)
                testPath.append(filePath + "/" + file + "/" + wav[j])
        lab += 1
    return wavPath, wavLabel, testPath, testLable


wavPath, wavLabel, testPath, testLabel = getwavPathAndwavLabel(filePath)
print("trainWavPath: ", len(wavPath))
print("testWavPath: ", len(testPath))
# '''按照相同的顺序打乱文件'''
# cc = list(zip(wavPath, wavLabel))
# random.shuffle(cc)
# wavPath[:], wavLabel[:] = zip(*cc)
# testPath = [len(wavPath)-1000 : len(wavPath)-1]
# testLabel = [len(wavPath)-1000 : len(wavPath)-1]

def getBW(batchSize=64, second=3, sampleRate=16000):
    """
    :param batchSize: 一个批次大小
    :param second: 音频的长度，默认3.5s,单位为sec
    :param sampleRate: 采样率
    :return:特征矩阵  和 标签
    """
    count = 0
    while True:

        '''按照相同的顺序打乱文件'''
        cc = list(zip(wavPath, wavLabel))
        random.shuffle(cc)
        wavPath[:], wavLabel[:] = zip(*cc)
        x = []
        y = []
        count = 0
        for index, wav in enumerate(wavPath):
            if count == batchSize:
                X = x
                Y = y
                # print(np.array(x).shape)
                X = np.array(X)  # (2, 64, 299, 3)
                Y = np.array(Y)
                Y = keras.utils.to_categorical(y, nClass)
                # print()
                x = []
                y = []
                count = 0

                yield [X, Y]
                # print(X.shape)
                # print(Y.shape)

            else:
                signal, srate = librosa.load(wav, sr=sampleRate)
                # 不符合条件
                if len(signal) < 3 * 16000:
                    continue
                # 归一化
                signal = signal / (max(np.abs(np.min(signal)), np.max(signal)))

                # 判断是否超过三秒，
                # 超过三秒则截断
                if len(signal) >= 3 * srate:
                    signal = signal[0:int(3 * srate)]
                # 少于三秒则填充0
                else:
                    signal = signal.tolist()
                    for j in range(3 * srate - len(signal)):
                        signal.append(0)
                    signal = np.array(signal)

                feat = psf.logfbank(signal[0:16000*3],samplerate=16000, nfilt=64)
                feat1 = psf.delta(feat, 1)
                feat2 = psf.delta(feat, 2)
                feat = feat.T[:, :, np.newaxis]
                feat1 = feat1.T[:, :, np.newaxis]
                feat2 = feat2.T[:, :, np.newaxis]
                fBank = np.concatenate((feat, feat1, feat2), axis=2)
                x.append(fBank)
                y.append(wavLabel[index])



                count +=1

def getTestBW(batchSize=64, second=3, sampleRate=16000):
    """
    :param batchSize: 一个批次大小
    :param second: 音频的长度，默认3.5s,单位为sec
    :param sampleRate: 采样率
    :return:特征矩阵  和 标签
    """
    count = 0
    while True:

        '''按照相同的顺序打乱文件'''
        cc = list(zip(testPath, testLabel))
        random.shuffle(cc)
        testPath[:], testLabel[:] = zip(*cc)
        x = []
        y = []
        count = 0
        for index, wav in enumerate(testPath):
            if count == batchSize:
                X = x
                Y = y
                # print(np.array(x).shape)
                X = np.array(X)  # (2, 64, 299, 3)
                Y = np.array(Y)
                Y = keras.utils.to_categorical(y, nClass)
                # print()
                x = []
                y = []
                count = 0

                yield [X, Y]
                # print(X.shape)
                # print(Y.shape)

            else:
                signal, srate = librosa.load(wav, sr=sampleRate)
                if len(signal) <3*16000:
                    continue
                # 归一化
                signal = signal / (max(np.abs(np.min(signal)), np.max(signal)))

                # 判断是否超过三秒，
                # 超过三秒则截断
                if len(signal) >= 3 * srate:
                    signal = signal[0:int(3 * srate)]
                # 少于三秒则填充0
                else:
                    signal = signal.tolist()
                    for j in range(3 * srate - len(signal)):
                        signal.append(0)
                    signal = np.array(signal)
                # print(len(signal))



                # feat = librosa.feature.mfcc(signal[0:16000*3], sr=16000, n_mfcc=64)
                feat = psf.logfbank(signal[0:16000*3],samplerate=16000, nfilt=64)
                # print("feat: ", feat.shape)
                feat1 = psf.delta(feat, 1)
                feat2 = psf.delta(feat, 2)
                feat = feat.T[:, :, np.newaxis]
                feat1 = feat1.T[:, :, np.newaxis]
                feat2 = feat2.T[:, :, np.newaxis]
                fBank = np.concatenate((feat, feat1, feat2), axis=2)
                x.append(fBank)
                y.append(testLabel[index])
                count +=1


if __name__ =="__main__":
    getTestBW(100)
    batchSize = 64
    # 卷积核个数
    nFilter = 64
    # 池化层的大小
    poolSize = [2, 2]
    # 池化层步长
    strideSize = [2, 2]
    # 卷积核的大小
    kernelSize = [5, 5]
    model = Sequential()
    model.add(Convolution2D(nFilter, (kernelSize[0], kernelSize[1]),
                            padding='same',
                            strides=(strideSize[0], strideSize[1]),
                            input_shape=(64, None, 3), name="cov1",
                            kernel_regularizer=keras.regularizers.l2()))
    # model.add(MaxPooling2D(pool_size=(poolSize[0], poolSize[1]), strides=(strideSize[0], strideSize[1]), padding="same", name="pool1"))

    # 将输入的维度按照给定模式进行重排
    model.add(Permute((2,1,3),name='permute'))
    # 该包装器可以把一个层应用到输入的每一个时间步上,GRU需要
    model.add(TimeDistributed(Flatten(),name='timedistrib'))

    # 三层GRU
    model.add(GRU(units=1024, return_sequences=True, name="gru1"))
    model.add(GRU(units=1024, return_sequences=True, name="gru2"))
    model.add(GRU(units=1024, return_sequences=True, name="gru3"))

    # temporal average
    def temporalAverage(x):
        return K.mean(x, axis=1)
    model.add(Lambda(temporalAverage, name="temporal_average"))

    # affine
    model.add(Dense(units=512, name="dense1"))

    # length normalization
    def lengthNormalization(x):
        return K.l2_normalize(x, axis=-1)
    model.add(Lambda(lengthNormalization, name="ln"))

    model.add(Dense(units=nClass ,name="dense2"))
    model.add(Activation("softmax"))

    sgd = Adam(lr=0.00001)

    model.compile(loss='categorical_crossentropy',
                  optimizer=sgd, metrics=['accuracy'])

    model.fit(getBW(batchSize, sampleRate=16000),steps_per_epoch = len(wavLabel)//batchSize, epochs=15)