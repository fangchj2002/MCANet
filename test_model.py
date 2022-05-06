
from glob import glob
import os
import numpy as np
import pandas as pd
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model
import constants as c
from pre_process import data_catalog, preprocess_and_save
from eval_metrics import evaluate
from models import fAlexnet, recurrent_model,Lenet,ResNet18,convolutional_model,dvector,TCM
from triplet_loss import deep_speaker_loss
from utils import get_last_checkpoint_if_any, create_dir_and_delete_content
import tensorflow as tf
from sklearn.metrics import roc_curve,auc
from tensorflow.keras.layers import Dense,Layer
from tensorflow.keras import regularizers
num_neg = c.TEST_NEGATIVE_No
class ArcLayer(Layer):
    def __init__(self, output_dim, s=30.0, m=0.2, regularizer=None, **kwargs):  # 初始化
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(ArcLayer, self).__init__(**kwargs)
        self.output_dim = output_dim
        self.s = s
        self.m = m
        self.W = None
        self.regularizer = regularizers.get(regularizer)

    def build(self, input_shape):  # 定义本层的权
        assert len(input_shape) >= 2
        input_dim = input_shape[-1]
        self.W = self.add_weight(name="kernel",
                                 shape=(input_dim, self.output_dim),
                                 initializer='glorot_uniform',
                                 regularizer=self.regularizer,
                                 trainable=True
                                 )
        self.bias = None
        self.built = True

    def call(self, inputs, **kwargs):  # 实现本层从输入张量到输出张量的计算图
        inputs = tf.nn.l2_normalize(inputs, 1, 1e-10)  # X 归一化
        self.W = tf.nn.l2_normalize(self.W, 0, 1e-10)  # W 归一化
        # cos(θ) --------------------------------------------------------------
        cos_theta = K.dot(inputs, self.W)
        # CosFace ====================== 余弦距离 =====================
        # phi = cos_theta - self.m
        # ArcFace ====================== 角度距离 =====================
        # controls the (theta + m) should in range [0, pi]
        theta = tf.acos(K.clip(cos_theta, -1.0+K.epsilon(), 1.0-K.epsilon()))
        phi = tf.cos(theta + self.m)
        # e^φ -----------------------------------------------------------------
        e_phi = K.exp(self.s * phi)
        e_cos = K.exp(self.s * cos_theta)
        # output
        output = e_phi / (e_phi + (K.sum(e_cos, axis=-1, keepdims=True)-e_cos))
        return output
def normalize_scores(m,epsilon=1e-12):
    return (m - np.mean(m)) / max(np.std(m),epsilon)

def clipped_audio(x, num_frames=c.NUM_FRAMES):
    if x.shape[0] > num_frames + 20:
        bias = np.random.randint(20, x.shape[0] - num_frames)
        clipped_x = x[bias: num_frames + bias]
    elif x.shape[0] > num_frames:
        bias = np.random.randint(0, x.shape[0] - num_frames)
        clipped_x = x[bias: num_frames + bias]
    else:
        clipped_x = x

    return clipped_x

def create_test_data(test_dir,check_partial):
    global num_neg
    libri = data_catalog(test_dir)
    unique_speakers = list(libri['speaker_id'].unique())
    np.random.shuffle(unique_speakers)
    num_triplets = len(unique_speakers)
    if check_partial:
        num_neg = 49; num_triplets = min(num_triplets, 30)
    test_batch = None
    for ii in range(num_triplets):
        anchor_positive_file = libri[libri['speaker_id'] == unique_speakers[ii]]
        if len(anchor_positive_file) <2:
            continue
        anchor_positive_file = anchor_positive_file.sample(n=2, replace=False)
        anchor_df = pd.DataFrame(anchor_positive_file[0:1])
        anchor_df['training_type'] = 'ancfrom thor'                      # 1 anchor，1 positive，num_neg negative
        if test_batch is None:
            test_batch = anchor_df.copy()
        else:
            test_batch = pd.concat([test_batch, anchor_df], axis=0)

        positive_df = pd.DataFrame(anchor_positive_file[1:2])
        positive_df['training_type'] = 'positive'
        test_batch = pd.concat([test_batch, positive_df], axis=0)

        negative_files = libri[libri['speaker_id'] != unique_speakers[ii]].sample(n=num_neg, replace=False)
        for index in range(len(negative_files)):
            negative_df = pd.DataFrame(negative_files[index:index+1])
            negative_df['training_type'] = 'negative'
            test_batch = pd.concat([test_batch, negative_df], axis=0)

    new_x = []
    for i in range(len(test_batch)):
        filename = test_batch[i:i + 1]['filename'].values[0]
        x = np.load(filename)
        new_x.append(clipped_audio(x))
    x = np.array(new_x)  # (batchsize, num_frames, 64, 1)
    new_y = np.hstack(([1], np.zeros(num_neg)))  # 1 positive, num_neg negative
    y = np.tile(new_y, num_triplets)
    return x, y

def batch_cosine_similarity(x1, x2):
    # https://en.wikipedia.org/wiki/Cosine_similarity
    # 1 = equal direction ; -1 = opposite direction
    mul = np.multiply(x1, x2)
    s = np.sum(mul,axis=1)

    #l1 = np.sum(np.multiply(x1, x1),axis=1)
    #l2 = np.sum(np.multiply(x2, x2), axis=1)
    # as values have have length 1, we don't need to divide by norm (as it is 1)
    return s

def call_similar(x):
    no_batch = int(x.shape[0] / (num_neg+2))  # each batch was consisted of 1 anchor ,1 positive , num_neg negative, so the number of batch
    similar = []
    for ep in range(no_batch):
        index = ep*(num_neg + 2)
        anchor = np.tile(x[index], (num_neg + 1, 1))
        pos_neg = x[index+1: index + num_neg + 2]
        sim = batch_cosine_similarity(anchor, pos_neg)
        similar.extend(sim)
    return np.array(similar)

def eval_model(model,train_batch_size=c.BATCH_SIZE * c.TRIPLET_PER_BATCH, test_dir= c.TEST_DIR, check_partial=False, gru_model=None) -> object:
    tf.random.set_seed(0)
    np.random.seed(0)
    x, y_true = create_test_data(test_dir,check_partial)
    batch_size = x.shape[0]
    b = x[0]
    num_frames = b.shape[0]

    '''
    print('test_data:')
    print('num_frames = {}'.format(num_frames))
    print('batch size: {}'.format(batch_size))
    print('input shape: {}'.format(input_shape))
    print('x.shape before reshape: {}'.format(x.shape))
    print('x.shape after  reshape: {}'.format(x.shape))
    print('y.shape: {}'.format(y_true.shape))
    '''
    #embedding = model.predict_on_batch(x)
    test_epoch = int(len(y_true)/train_batch_size)
    embedding = None
    for ep in range(test_epoch):
        x_ = x[ep*train_batch_size: (ep + 1)*train_batch_size]
        embed = model.predict_on_batch(x_)
        if embedding is None:
            embedding = embed[:]
        else:
            embedding = np.concatenate([embedding, embed], axis=0)
    y_pred = call_similar(embedding)
    if gru_model is not None:
        embedding_gru = None
        for ep in range(test_epoch):
            x_ = x[ep * train_batch_size: (ep + 1) * train_batch_size]
            embed = model.predict_on_batch(x_)
            if embedding_gru is None:
                embedding_gru = embed.copy()
            else:
                embedding_gru = np.concatenate([embedding_gru, embed], axis=0)
        y_pred_gru = call_similar(embedding_gru)

        y_pred = (normalize_scores(y_pred) + normalize_scores(y_pred_gru))/2  # or   y_pred = (y_pred + y_pred_gru)/2

    nrof_pairs = min(len(y_pred), len(y_true))
    y_pred = y_pred[:nrof_pairs]
    y_true = y_true[:nrof_pairs]
    fm, tpr,acc, eer = evaluate(y_pred, y_true)
    return fm, tpr, acc, eer
  #  fpr = dict()
  #  tpr = dict()
  #  roc_auc = dict()
 #  for i in range(40):
  #      fpr[i], tpr[i], _ = roc_curve(y_true[:], y_pred[:])
  #      roc_auc[i] = auc(fpr[i], tpr[i])
  #  fpr["micro"], tpr["micro"], _ = roc_curve(y_true.ravel(), y_pred.ravel())
  #  roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
  #  return fm, fpr["micro"],tpr["micro"],roc_auc["micro"],  acc, eer


if __name__ == '__main__':
    model = TCM()
    gru_model = None
    last_checkpoint = get_last_checkpoint_if_any(c.PRE_CHECKPOINT_FOLDER)
    if last_checkpoint is not None:
        print('Found checkpoint [{}]. Resume from here...'.format(last_checkpoint))
       # layerName = "dense4"
       # model = Model(inputs=model.input, outputs=model.get_layer(layerName).output)
       # model.load_weights(last_checkpoint)
        x = model.output
      #  x = Dense(1211, activation='softmax', name='softmax_layer')(x)
        x = Dense(1211)(x)
       # x = ArcLayer(1211, s=30.0, m=0.1)(x)
        pre_model = Model(model.input, x)
        pre_model.load_weights(last_checkpoint)

    if c.COMBINE_MODEL:
        gru_model = recurrent_model()
        last_checkpoint = get_last_checkpoint_if_any(c.PRE_CHECKPOINT_FOLDER)
        if last_checkpoint is not None:
            print('Found checkpoint [{}]. Resume from here...'.format(last_checkpoint))
            gru_model.load_weights(last_checkpoint)


 #   fm,fpr,tpr,auc,acc,eer= eval_model(model, check_partial=False,gru_model=None),use_bias=False
 #   print("f-measure = {0},FPR={2},TPR= {1},AUC={3},accuracy = {4}, equal error rate = {5}".format(fm, fpr,tpr,auc, acc, eer))
    fm, tpr, acc, eer = eval_model(model, check_partial=True, gru_model=None)
    print("f-measure = {0},TPR= {1},accuracy = {2}, equal error rate = {3}".format(fm,tpr,acc, eer))
