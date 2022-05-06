import tensorflow as tf
from tensorflow.keras import backend as K
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
# reference:
# https://github.com/hao-qiang/AM-Softmax/blob/master/AM-Softmax.ipynb
def amsoftmax_loss(y_true, y_pred, scale=30.0, margin=0.35):
    # make two constant tensors.
    m = K.constant(margin, name='m')
    s = K.constant(scale, name='s')
    # reshape the label
    label = K.reshape(K.argmax(y_true, axis=-1), shape=(-1, 1))
    label = K.cast(label, dtype=tf.int32)

    pred_batch = K.reshape(tf.range(K.shape(y_pred)[0]), shape=(-1, 1))
    # concat the two column vectors, one is the pred_batch, the other is label.
    ground_truth_indices = tf.concat([pred_batch,
                                      K.reshape(label, shape=(-1, 1))], axis=1)
    # get ground truth scores by indices
    ground_truth_scores = tf.gather_nd(y_pred, ground_truth_indices)

    # if ground_truth_score > m, group_truth_score = group_truth_score - m
    added_margin = K.cast(K.greater(ground_truth_scores, m),
                          dtype=tf.float32) * m
    added_margin = K.reshape(added_margin, shape=(-1, 1))
    added_embedding_feature = tf.subtract(y_pred, y_true * added_margin) * s

    cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_true,
                                                               logits=added_embedding_feature)
    loss = tf.reduce_mean(cross_entropy)
    return loss

