
# train-clean-100: 251 speaker, 28539 utterance
# train-clean-360: 921 speaker, 104104 utterance. Extract audio features and save it as npy file, cost 8443.623185157776 seconds
# test-clean: 40 speaker, 2620 utterance
# batchisize 32*3 : train on triplet: 3.3s/steps , softmax pre train: 3.1 s/steps

from models import fAlexnet,Lenet,ResNet18,convolutional_model,Alexnet,recurrent_model,VGG_M,dvector,TCM,Res2Net
from test_model import eval_model
import logging
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense,Layer,InputSpec,Lambda,LeakyReLU
from tensorflow.keras.optimizers import Adam
import numpy as np
import random
import constants as c
import utils
from tensorflow.keras import regularizers,initializers, constraints
from pre_process import data_catalog, preprocess_and_save
from select_batch import clipped_audio
from time import time
import sys
from sklearn.model_selection import train_test_split
from tensorflow.keras import backend as K
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
kls = tf.keras.losses
k = tf.keras

EETA_DEFAULT = 0.001

class SparseAmsoftmaxLoss(kls.Loss):

  def __init__(self,
               scale: int = 30,
               margin: int = 0.35,
               batch_size: int = None,
               reduction='auto',
               name=None):
    """ sparse addivate margin softmax
        Parameters
        ----------
        scale : int, optional
            by default 30
        margin : int, optional
            by default 0.35
    """
    super().__init__(reduction=reduction, name=name)
    self.scale = scale
    self.margin = margin
    if batch_size:
      self.batch_idxs = tf.expand_dims(
          tf.range(0, batch_size, dtype=tf.int32), 1)  # shape [batch,1]

  def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    idxs = tf.concat([self.batch_idxs, tf.cast(y_true, tf.int32)], 1)
    y_true_pred = tf.gather_nd(y_pred, idxs)
    y_true_pred = tf.expand_dims(y_true_pred, 1)
    y_true_pred_margin = y_true_pred - self.margin
    _Z = tf.concat([y_pred, y_true_pred_margin], 1)
    _Z = _Z * self.scale
    logZ = tf.math.reduce_logsumexp(_Z, 1, keepdims=True)
    logZ = logZ + tf.math.log(1 - tf.math.exp(self.scale * y_true_pred - logZ))
    return -y_true_pred_margin * self.scale + logZ


class AmsoftmaxLoss(SparseAmsoftmaxLoss):

  def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    y_pred = (y_true * (y_pred - self.margin) + (1 - y_true) * y_pred) * self.scale
    return tf.nn.softmax_cross_entropy_with_logits(y_true, y_pred)

# Loss


def loadFromList(x_paths, batch_start, limit, labels_to_id, no_of_speakers ):
    x1 = []
    y_ = []
    for i in range(batch_start, limit):
        x1_ = np.load(x_paths[i])
        x1.append(clipped_audio(x1_))
        last = x_paths[i].split("/")[-1]
        y_.append(labels_to_id[last.split("-")[0]])
    x1= np.asarray(x1)
    y = np.eye(no_of_speakers)[y_]    #one-hot
    y = np.asarray(y)
    return x1,y

def batchTrainingImageLoader(train_data, labels_to_id, no_of_speakers, batch_size=c.BATCH_SIZE * c.TRIPLET_PER_BATCH):
    paths = train_data
    L = len(paths)
    while True:
        np.random.shuffle(paths)
        batch_start = 0
        batch_end = batch_size

        while batch_end <= L:
            x1_train_t,y_train_t = loadFromList(paths, batch_start, batch_end, labels_to_id, no_of_speakers)
            randnum = random.randint(0, 100)
            random.seed(randnum)
            random.shuffle(x1_train_t)
            random.seed(randnum)
            random.shuffle(y_train_t)
            yield  x1_train_t,y_train_t
            batch_start += batch_size
            batch_end += batch_size

def batchTestImageLoader(test_data, labels_to_id, no_of_speakers, batch_size=c.BATCH_SIZE * c.TRIPLET_PER_BATCH):
    paths = test_data
    L = len(paths)
    while True:
        np.random.shuffle(paths)
        batch_start = 0
        batch_end = batch_size

        while batch_end <= L:
            x1_test_t, y_test_t = loadFromList(paths, batch_start, batch_end, labels_to_id, no_of_speakers)
          #  yield ({"main_input":x1_test_t,'aux_input':x2_test_t}, {"ln":y_test_t})
            yield x1_test_t, y_test_t
            batch_start += batch_size
            batch_end += batch_size

def split_data(files1,labels, batch_size):
    test_size = max(batch_size/len(labels),0.2)
    train_paths, test_paths, y_train, y_test = train_test_split(files1,labels,test_size=test_size, random_state=42)
    return train_paths, test_paths

def amsoftmax_loss(y_true, y_pred, scale=30, margin=0.35):
    y_pred = y_true * (y_pred - margin) + (1 - y_true) * y_pred
    y_pred *= scale
    return K.categorical_crossentropy(y_true, y_pred, from_logits=True)

class CircleLoss(kls.Loss):

  def __init__(self,
               gamma: int = 256,
               margin: float = 0.25,
               batch_size: int = None,
               reduction='auto',
               name=None):
    super().__init__(reduction=reduction, name=name)
    self.gamma = gamma
    self.margin = margin
    self.O_p = 1 + self.margin
    self.O_n = -self.margin
    self.Delta_p = 1 - self.margin
    self.Delta_n = self.margin
    if batch_size:
      self.batch_size = batch_size
      self.batch_idxs = tf.expand_dims(
          tf.range(0, batch_size, dtype=tf.int32), 1)  # shape [batch,1]

  def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """ NOTE : y_pred must be cos similarity
    Args:
        y_true (tf.Tensor): shape [batch,ndim]
        y_pred (tf.Tensor): shape [batch,ndim]
    Returns:
        tf.Tensor: loss
    """
    alpha_p = tf.nn.relu(self.O_p - tf.stop_gradient(y_pred))
    alpha_n = tf.nn.relu(tf.stop_gradient(y_pred) - self.O_n)
    # yapf: disable
    y_true = tf.cast(y_true, tf.float32)
    y_pred = (y_true * (alpha_p * (y_pred - self.Delta_p)) +
              (1 - y_true) * (alpha_n * (y_pred - self.Delta_n))) * self.gamma
    # yapf: enable
    return tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=y_pred)

class ProxyAnchorLoss(kls.Loss):

  def __init__(self,
               gamma: int = 32,
               margin: float = 0.1,
               batch_size: int = 64,
               reduction='auto',
               name=None):
    super().__init__(reduction=reduction, name=name)
    self.gamma = gamma
    self.margin = margin
    if batch_size:
      self.batch_size = batch_size
      self.batch_idxs = tf.expand_dims(
          tf.range(0, batch_size, dtype=tf.int32), 1)  # shape [batch,1]

  def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """ NOTE : y_pred must be cos similarity
    Args:
        y_true (tf.Tensor): shape [batch,ndim]
        y_pred (tf.Tensor): shape [batch,ndim]
    Returns:
        tf.Tensor: loss
    """
    # The number of positive proxies
    num_valid_proxies = tf.reduce_sum(tf.cast(tf.reduce_sum(
        y_true, 0, keepdims=True) != 0, tf.float32))

    # yapf: disable
    y_pred = ((y_true * (y_pred - self.margin) / num_valid_proxies) +
              ((1 - y_true) * (y_pred - self.margin) / tf.cast(tf.shape(y_true)[-1], tf.float32))) * self.gamma
    # yapf: enable
    return tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=y_pred)



def focal_loss(gamma=2., alpha=.25):
	def focal_loss_fixed(y_true, y_pred):
		pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
		pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
		return -K.mean(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1+K.epsilon())) - K.mean((1 - alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0 + K.epsilon()))
	return focal_loss_fixed

def powernorm(x):
    power=K.sum(x**2)
    power_sqr=K.sqrt(power)
    normalized=x/power_sqr
    return normalized
def my_loss(y_true,y_pred):
    y_true_nor=powernorm(y_true)
    y_pred_nor=powernorm(y_pred)
    return K.mean(K.square(y_pred_nor - y_true_nor),axis = -1)

class ArcLayer(Layer):
    def __init__(self, output_dim, s=30.0, m=0.1, regularizer=None, **kwargs):  # 初始化
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
        phi = K.cos(theta + self.m)
        # e^φ -----------------------------------------------------------------
        e_phi = K.exp(self.s * phi)
        e_cos = K.exp(self.s * cos_theta)
        # output
        output = e_phi / (e_phi + (K.sum(e_cos, axis=-1, keepdims=True)-e_cos))
        return output

    def compute_output_shape(self, input_shape):  # 指定输入及输出张量形状变化的逻辑!
        return input_shape[0], self.output_dim

def AAMsoftmax_loss(y_true, y_pred):
    loss = -K.mean(K.log(K.clip(K.sum(y_true * y_pred, axis=-1), K.epsilon(), None)), axis=-1)
    return loss

def main():
    import argparse
    parser = argparse.ArgumentParser()
    # set up training configuration.
    parser.add_argument('--gpu', default='', type=str)
    parser.add_argument('--resume', default='', type=str)
    parser.add_argument('--batch_size', default=64, type=int)
   # parser.add_argument('--data_path', default='/scratch/local/ssd/weidi/voxceleb2/dev/wav', type=str)
  #  parser.add_argument('--multiprocess', default=12, type=int)
    # set up network configuration.
    parser.add_argument('--net', default='resnet34s', choices=['resnet34s', 'resnet34l'], type=str)
    parser.add_argument('--ghost_cluster', default=2, type=int)
    parser.add_argument('--vlad_cluster', default=10, type=int)
    parser.add_argument('--bottleneck_dim', default=512, type=int)
    parser.add_argument('--aggregation_mode', default='gvlad', choices=['avg', 'vlad', 'gvlad'], type=str)
    # set up learning rate, training loss and optimizer.
   # parser.add_argument('--epochs', default=56, type=int)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--warmup_ratio', default=0, type=float)
    parser.add_argument('--loss', default='softmax', choices=['softmax', 'amsoftmax'], type=str)
 #   parser.add_argument('--optimizer', default='adam', choices=['adam', 'sgd'], type=str)
    parser.add_argument('--ohem_level', default=0, type=int,
                        help='pick hard samples from (ohem_level * batch_size) proposals, must be > 1')
    global args
    args = parser.parse_args()

    batch_size = c.BATCH_SIZE * c.TRIPLET_PER_BATCH
    train_path = c.DATASET_DIR
    libri = data_catalog(train_path)
    files1 = list(libri['filename'])
    labels= list(libri['speaker_id'])

    labels_to_id = {}
    id_to_labels = {}
    i = 0

    for label in np.unique(labels):#labels=27,labels1是27个标签为19的演讲者
        labels_to_id[label] = i
        id_to_labels[i] = label
        i += 1

    no_of_speakers = len(np.unique(labels))

    train_data, test_data = split_data(files1,labels, batch_size)
    batchloader = batchTrainingImageLoader(train_data,labels_to_id,no_of_speakers, batch_size=batch_size)
    testloader = batchTestImageLoader(test_data, labels_to_id, no_of_speakers, batch_size=batch_size)

    test_steps = int(len(test_data)/batch_size)
    x1_test, y_test= testloader.__next__()
    b = x1_test[0]
    num_frames = b.shape[0]
    logging.info('num_frames = {}'.format(num_frames))
    logging.info('batch size: {}'.format(batch_size))
    logging.info("x1_shape:{0}, y_shape:{1}".format(x1_test.shape, y_test.shape))
    print(x1_test.shape)
    print(y_test.shape)

  #  model = vmodel.vggvox_resnet2d_icassp(input_dim=(num_frames, 64, 1),
                                       #    num_class=240,
                                       #    mode='train', args=args)
    base_model =Res2Net(input_shape=(num_frames,64, 1),batch_size=batch_size ,num_frames=num_frames)

    x = base_model.output
   # x = ArcLayer(240, s=30.0,m=0.1)(x)
  #  layerName = "dense4"
 #   model = Model(inputs=model.input, outputs=model.get_layer(layerName).output)
  #  x = Dense(no_of_speakers, activation='softmax', name='softmax_layer')(x)
    x = Dense(no_of_speakers)(x)
    #x = ArcFace( n_classes=240, s=30.0, m=0.50)
 #   arcface_output = af_layer(x)

    model = Model(base_model.input,x)
    print(x.shape)
    logging.info(model.summary())
   # loss = OSM_CAA_Loss()
  #  osm_loss = loss.forwardnp.float32))
    #     #    #
   # w = tf.get_variable("w", initializer=np.array([10], dtype= b = tf.get_variable("b", initializer=np.array([-5], dtype=np.float32))
   # embedded = model.outputs  # the last ouput is the embedded d-vector
   # sim_matrix = similarity(embedded, w, b)
 #   print("similarity matrix size: ", sim_matrix.shape)
    exponential_decay = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=0.001,decay_steps=500, decay_rate=0.95)
  #  adam_wn = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07)
  #  model.compile(optimizer=Adam(exponential_decay), loss=NTXentLoss(temperature=1.0), metrics=['accu\\racy'])
   # model.compile(optimizer=Adam(exponential_decay),loss="categorical_crossentropy",metrics=['accuracy'])
   # model.fit((x1_train,x2_train),y_train,epochs=2, batch_size=batch_size,validation_data=((x1_test, x2_test),y_test))
   # model.compile(optimizer=Adam(exponential_decay), loss="categorical_crossentropy", metrics=['accuracy'])
   # model.fit((x_train,y_train), epochs=30, batch_size=batch_size, validation_data=(x1_test,y_test))
    model.compile(optimizer=Adam(exponential_decay), loss=amsoftmax_loss, metrics=['accuracy'])
    print("printing format per batch:", model.metrics_names)
    # y_ = np.argmax(y_train, axis=0)
    # class_weights = sklearn.utils.class_weight.compute_class_weight('balanced', np.unique(y_), y_)
    grad_steps = 0
    last_checkpoint = utils.get_last_checkpoint_if_any(c.PRE_CHECKPOINT_FOLDER)
    #last_checkpoint = None
    if last_checkpoint is not None:
        logging.info('Found checkpoint [{}]. Resume from here...'.format(last_checkpoint))
        model.load_weights(last_checkpoint)
        grad_steps = int(last_checkpoint.split('_')[-2])
        logging.info('[DONE]')
    orig_time = time()
    Num_Iter=10020
    current_iter = 1
    while current_iter <Num_Iter:
        current_iter += 1
        orig_time = time()
        x_train, y_train = batchloader.__next__()
        [loss, acc] = model.train_on_batch(x_train, y_train)
      #  logging.info('Train Steps:{0}, Time:{1:.2f}s, Loss={2}, Accuracy={3},'.format(grad_steps, time() - orig_time, loss,acc))
        with open(c.PRE_CHECKPOINT_FOLDER + "/train_loss_acc.txt", "a") as f:
            f.write("{0},{1},{2}\n".format(grad_steps, loss, acc))

        if grad_steps % 20 == 0:
            losses = [];accs = []
            for ss in range(test_steps):
                [loss,acc] = model.test_on_batch(x1_test, y_test)
                x1_test, y_test = testloader.__next__()
                losses.append(loss); accs.append(acc)
            loss = np.mean(np.array(losses)); acc = np.mean(np.array(accs))
            logging.info('Test the Data ---------- Steps:{0}, Loss={1}, Accuracy={2}, '.format(grad_steps,loss,acc))
            with open(c.PRE_CHECKPOINT_FOLDER + "/test_loss_acc.txt", "a") as f:
                f.write("{0},{1},{2}\n".format(grad_steps, loss, acc))

        if grad_steps  % c.SAVE_PER_EPOCHS == 0:
            utils.create_dir_and_delete_content(c.PRE_CHECKPOINT_FOLDER)
            model.save_weights('{0}/model_{1}_{2:.5f}.h5'.format(c.PRE_CHECKPOINT_FOLDER, grad_steps, loss))


        grad_steps += 1

'''
if __name__ == '__main__'的意思是：当.py文件被直接运行时，if __name__ == '__main__'之下的代码块将被运行；
当.py文件以模块形式被导入时，if __name__ == '__main__'之下的代码块不被运行。
'''
if __name__ == '__main__':
    logging.basicConfig(handlers=[logging.StreamHandler(stream=sys.stdout)], level=logging.INFO,
                        format='%(asctime)-15s [%(levelname)s] %(filename)s/%(funcName)s | %(message)s')
    main()
