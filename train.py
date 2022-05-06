
# train-clean-100: 251 speaker, 28539 utterance
# train-clean-360: 921 speaker, 104104 utterance
# test-clean: 40 speaker, 2620 utterance
# merged test: 80 speaker, 5323 utterance
# batchisize 32*3 : train on triplet: 5s - > 3.1s/steps , softmax pre train: 3.1 s/steps


import logging
from time import time
import numpy as np
import sys
import os
import random
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
import constants as c
import select_batch
from pre_process import data_catalog, preprocess_and_save
from models import fAlexnet, recurrent_model,Lenet,ResNet18,convolutional_model,Alexnet
from random_batch import stochastic_mini_batch
from triplet_loss import deep_speaker_loss
from utils import get_last_checkpoint_if_any, create_dir_and_delete_content
from test_model import eval_model
import tensorflow as tf
from pretraining import AAMsoftmax_loss,amsoftmax_loss
from tensorflow.keras.layers import Dense,Layer
from tensorflow.keras import backend as K
from tensorflow.keras import regularizers

def create_dict(files,labels,spk_uniq):
    train_dict = {}
    for i in range(len(spk_uniq)):
        train_dict[spk_uniq[i]] = []

    for i in range(len(labels)):
        train_dict[labels[i]].append(files[i])

    for spk in spk_uniq:
        if len(train_dict[spk]) < 2:
            train_dict.pop(spk)
    unique_speakers=list(train_dict.keys())
    return train_dict, unique_speakers

class ArcFace(Layer):
    def __init__(self, n_classes=1211, s=30.0, m=0.50, regularizer=None, **kwargs):
        super(ArcFace, self).__init__(**kwargs)
        self.n_classes = n_classes
        self.s = s
        self.m = m
        self.regularizer = regularizers.get(regularizer)

    def build(self, input_shape):
        super(ArcFace, self).build(input_shape[0])
        self.W = self.add_weight(name='W',
                                shape=(input_shape[0][-1], self.n_classes),
                                initializer='glorot_uniform',
                                trainable=True,
                                regularizer=self.regularizer)

    def call(self, inputs):
        x, y = inputs
        c = K.shape(x)[-1]
        # normalize feature
        x = tf.nn.l2_normalize(x, axis=1)
        # normalize weights
        W = tf.nn.l2_normalize(self.W, axis=0)
        # dot product
        logits = x @ W
        # add margin
        # clip logits to prevent zero division when backward
        theta = tf.acos(K.clip(logits, -1.0 + K.epsilon(), 1.0 - K.epsilon()))
        target_logits = tf.cos(theta + self.m)
        # sin = tf.sqrt(1 - logits**2)
        # cos_m = tf.cos(logits)
        # sin_m = tf.sin(logits)
        # target_logits = logits * cos_m - sin * sin_m
        #
        logits = logits * (1 - y) + target_logits * y
        # feature re-scale
        logits *= self.s
        out = tf.nn.softmax(logits)

        return out

    def compute_output_shape(self, input_shape):
        return (None, self.n_classes)

def main(libri_dir=c.DATASET_DIR):
    #config = ConfigProto()
   # config.gpu_options.allow_growth = True
   # session = InteractiveSession(config=config)
    PRE_TRAIN = c.PRE_TRAIN
    logging.info('Looking for fbank features [.npy] files in {}.'.format(libri_dir))
    libri = data_catalog(libri_dir)

    if len(libri) == 0:
        logging.warning('Cannot find npy files, we will load audio, extract features and save it as npy file')
        logging.warning('Waiting for preprocess...')
        preprocess_and_save(c.WAV_DIR, c.DATASET_DIR)
        libri = data_catalog(libri_dir)
        if len(libri) == 0:
            logging.warning('Have you converted flac files to wav? If not, run audio/convert_flac_2_wav.sh')
            exit(1)
    unique_speakers = libri['speaker_id'].unique()
    spk_utt_dict, unique_speakers = create_dict(libri['filename'].values,libri['speaker_id'].values,unique_speakers)
    select_batch.create_data_producer(unique_speakers, spk_utt_dict)

    batch = stochastic_mini_batch(libri, batch_size=c.BATCH_SIZE, unique_speakers=unique_speakers)
    batch_size = c.BATCH_SIZE
    x, y = batch.to_inputs()
   # b = x1[0]
    d = x[0]
    num_frames = d.shape[0]
    train_batch_size = batch_size
    #batch_shape = [batch_size * num_frames] + list(b.shape[1:])  # A triplet has 3 parts.
   # input_shape = (num_frames, b.shape[1], b.shape[2])
    another_shape=(d.shape[0],d.shape[1],d.shape[2])

    logging.info('num_frames = {}'.format(num_frames))
    logging.info('batch size: {}'.format(batch_size))
    logging.info('another shape :{0}'.format(another_shape))
    logging.info('x1.shape : {0}'.format(x.shape))
    orig_time = time()
    model =fAlexnet(batch_size=batch_size, input_shape=(num_frames, 64, 1), num_frames=num_frames)

    logging.info(model.summary())
    gru_model = None
    if c.COMBINE_MODEL:
        gru_model = recurrent_model(input_shape=another_shape, batch_size=batch_size, num_frames=num_frames)
        logging.info(gru_model.summary())
    grad_steps = 0

    if PRE_TRAIN:
        last_checkpoint = get_last_checkpoint_if_any(c.PRE_CHECKPOINT_FOLDER)
        if last_checkpoint is not None:
            logging.info('Found pre-training checkpoint [{}]. Resume from here...'.format(last_checkpoint))
            x = model.output
          #  x = ArcLayer(len(unique_speakers), s=30.0, m=0.1)(x)
            x = Dense(len(unique_speakers),name='AMsoftmax_layer')(x)
            pre_model = Model(model.input, x)
          #  pre_model = model
            pre_model.load_weights(last_checkpoint)
            grad_steps = int(last_checkpoint.split('_')[-2])
            logging.info('Successfully loaded pre-training model')

    else:
        last_checkpoint = get_last_checkpoint_if_any(c.CHECKPOINT_FOLDER)
        if last_checkpoint is not None:
            logging.info('Found checkpoint [{}]. Resume from here...'.format(last_checkpoint))
            model.load_weights(last_checkpoint)
            grad_steps = int(last_checkpoint.split('_')[-2])
            logging.info('[DONE]')
        if c.COMBINE_MODEL:
            last_checkpoint = get_last_checkpoint_if_any(c.GRU_CHECKPOINT_FOLDER)
            if last_checkpoint is not None:
                logging.info('Found checkpoint [{}]. Resume from here...'.format(last_checkpoint))
                gru_model.load_weights(last_checkpoint)
                logging.info('[DONE]')

    #adam = Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    exponential_decay = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=0.001, decay_steps=500, decay_rate=0.95)
    model.compile(optimizer=Adam(exponential_decay), loss=deep_speaker_loss)
    if c.COMBINE_MODEL:
        gru_model.compile(optimizer='adam', loss=deep_speaker_loss)
    print("model_build_time",time()-orig_time)
    logging.info('Starting training...')
    lasteer = 10
    eer = 1
    
    Num_Iter = 30020
    current_iter = 1
    while current_iter <Num_Iter:
        current_iter += 1
        orig_time = time()
      #  x, _ = batch.to_inputs()
        x, _ = select_batch.best_batch(model, batch_size=c.BATCH_SIZE)
        print("select_batch_time:", time() - orig_time)
        y = np.random.uniform(size=(x.shape[0], 1))
        # If "ValueError: Error when checking target: expected ln to have shape (None, 512) but got array with shape (96, 1)"
        # please modify line 121 to following line
        # y = np.random.uniform(size=(x.shape[0], 512))
        logging.info('== Presenting step #{0}'.format(grad_steps))
        orig_time = time()
        if (grad_steps) % 20 == 0:
           loss = model.train_on_batch(x, y)
           test_loss = model.test_on_batch(x, y)
           logging.info('== Processed in {0:.2f}s by the network, testing loss = {1}'.format(time() - orig_time, test_loss))
           logging.info('== Processed in {0:.2f}s by the network, training loss = {1}'.format(time() - orig_time, loss))
           if c.COMBINE_MODEL:
               loss1 = gru_model.train_on_batch(x, y)
               logging.info( '== Processed in {0:.2f}s by the gru-network, training loss = {1:.4f}.'.format(time() - orig_time, loss1))
               with open(c.GRU_CHECKPOINT_FOLDER + '/losses_gru.txt', "a") as f:
                    f.write("{0},{1}\n".format(grad_steps, loss1))
        # record training loss
           with open(c.LOSS_LOG, "a") as f:
              f.write("{0},{1}\n".format(grad_steps, loss))
           with open(c.Test_LOSS_LOG, "a") as f:
              f.write("{0},{1}\n".format(grad_steps, test_loss))

        if (grad_steps) % 20 == 0:
            fm1, tpr1, acc1, eer1 = eval_model(model, train_batch_size, test_dir=c.DATASET_DIR, check_partial=True,gru_model=None)
            logging.info('train data EER = {0:.3f}, F-measure = {1:.3f}, Accuracy = {2:.3f} '.format(eer1, fm1, acc1))
            with open(c.CHECKPOINT_FOLDER + '/train_acc_eer.txt', "a") as f:
                f.write("{0},{1},{2},{3}\n".format(grad_steps, eer1, fm1, acc1))

        if (grad_steps) % 20 == 0:
            fm, tpr, acc, eer = eval_model(model, train_batch_size, test_dir=c.TEST_DIR, gru_model=None)
            logging.info('== Testing model after batch #{0}'.format(grad_steps))
            logging.info('EER = {0:.3f}, F-measure = {1:.3f}, Accuracy = {2:.3f} '.format(eer, fm, acc))
            with open(c.TEST_LOG, "a") as f:
                f.write("{0},{1},{2},{3}\n".format(grad_steps, eer, fm, acc))
        # checkpoints are really heavy so let's just keep the last one.
        if (grad_steps ) % c.SAVE_PER_EPOCHS == 0:
            create_dir_and_delete_content(c.CHECKPOINT_FOLDER)
            model.save_weights('{0}/model_{1}_{2:.5f}.h5'.format(c.CHECKPOINT_FOLDER, grad_steps, loss))
            if c.COMBINE_MODEL:
                gru_model.save_weights('{0}/grumodel_{1}_{2:.5f}.h5'.format(c.GRU_CHECKPOINT_FOLDER, grad_steps, loss1))
            if eer < lasteer:
                files = sorted(filter(lambda f: os.path.isfile(f) and f.endswith(".h5"),
                                      map(lambda f: os.path.join(c.BEST_CHECKPOINT_FOLDER, f), os.listdir(c.BEST_CHECKPOINT_FOLDER))),
                               key=lambda file: file.split('/')[-1].split('.')[-2], reverse=True)
                lasteer = eer
                for file in files[:-4]:
                    logging.info("removing old model: {}".format(file))
                    os.remove(file)
                model.save_weights(c.BEST_CHECKPOINT_FOLDER+'/best_model{0}_{1:.5f}.h5'.format(grad_steps, eer))
                if c.COMBINE_MODEL:
                    files = sorted(filter(lambda f: os.path.isfile(f) and f.endswith(".h5"),
                                          map(lambda f: os.path.join(c.BEST_CHECKPOINT_FOLDER, f),
                                              os.listdir(c.BEST_CHECKPOINT_FOLDER))),
                                   key=lambda file: file.split('/')[-1].split('.')[-2], reverse=True)
                    lasteer = eer
                    for file in files[:-4]:
                        logging.info("removing old model: {}".format(file))
                        os.remove(file)
                    gru_model.save_weights(c.BEST_CHECKPOINT_FOLDER+'/best_gru_model{0}_{1:.5f}.h5'.format(grad_steps, eer))

        grad_steps += 1

if __name__ == '__main__':
    logging.basicConfig(handlers=[logging.StreamHandler(stream=sys.stdout)], level=logging.INFO,
                        format='%(asctime)-15s [%(levelname)s] %(filename)s/%(funcName)s | %(message)s')
    main()
