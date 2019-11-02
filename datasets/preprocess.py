import tensorflow as tf
import time
import logging
import numpy as np
import os

logger = logging.getLogger('preprocess')


def preprocess_dataset(x, x_test, y, y_test):
    # One-hot
    logger.debug(f'One hot: shape of y before: {y.shape}')
    y = one_hot(y)
    logger.debug(f'One hot: shape of y after: {y.shape}')
    logger.debug(f'One hot: shape of y_test before: {y_test.shape}')
    y_test = one_hot(y_test)
    logger.debug(f'One hot: shape of y_test after: {y_test.shape}')

    # ZCA whitening
    logger.info("ZCA whitening")
    time1 = time.time()
    x, x_test = zca(x, x_test)
    time2 = time.time()
    logger.info(f'Time Elapsed - ZCA Whitening: {time2 - time1}')

    return x, x_test, y, y_test


def preprocess_dataset_and_save(x, x_test, y, y_test, data_directory):
    x, x_test, y, y_test = preprocess_dataset(x, x_test, y, y_test)
    x_np = np.array(x)
    y_np = np.array(y)
    x_test_np = np.array(x_test)
    y_test_np = np.array(y_test)
    os.makedirs('preprocessed_data')
    np.save(data_directory + '/preprocessed_data/x', x_np)
    np.save(data_directory + '/preprocessed_data/y', y_np)
    np.save(data_directory + '/preprocessed_data/x_test', x_test_np)
    np.save(data_directory + '/preprocessed_data/y_test', y_test_np)
    return x, x_test, y, y_test


def load_preprocessed_data(data_directory):
    breakpoint()
    x = np.load(data_directory + '/preprocessed_data/x.npy')
    x = tf.convert_to_tensor(x)
    y = np.load(data_directory + '/preprocessed_data/y.npy')
    y = tf.convert_to_tensor(y)
    x_test = np.load(data_directory + '/preprocessed_data/x_test.npy')
    x_test = tf.convert_to_tensor(x_test)
    y_test = np.load(data_directory + '/preprocessed_data/y_test.npy')
    y_test = tf.convert_to_tensor(y_test)
    return x, x_test, y, y_test


###############################################################################
#    Title: ZCA
###############################################################################
#    Description:
#        This function applies ZCA Whitening to the image set
#
#    Parameters:
#        x_1           Array of MxNxC images to compute the ZCA Whitening
#        x_2           Array of MxNxC images to apply the ZCA transform
#        num_batch    Number of batches to do the computation
#
#    Returns:
#        An array of MxNxC zca whitened images
###############################################################################
@tf.function
def zca(x_1, x_2, epsilon=1e-5):
    with tf.name_scope('ZCA'):
        flatx = tf.cast(tf.reshape(
            x_1, (-1, np.prod(x_1.shape[-3:])), name="reshape_flat"),
            tf.float64, name="flatx")
        sigma = tf.tensordot(tf.transpose(flatx), flatx, 1, name="sigma") / \
            tf.cast(tf.shape(flatx)[0], tf.float64)  # N-1 or N?
        s, u, v = tf.linalg.svd(sigma, name="svd")
        pc = tf.tensordot(tf.tensordot(u, tf.linalg.diag(
            1. / tf.math.sqrt(s+epsilon)), 1, name="inner_dot"),
            tf.transpose(u), 1, name="pc")

        net1 = tf.tensordot(flatx, pc, 1, name="whiten1")
        net1 = tf.reshape(net1, np.shape(x_1), name="output1")

        flatx2 = tf.cast(tf.reshape(
            x_2, (-1, np.prod(x_2.shape[-3:])), name="reshape_flat2"),
            tf.float64, name="flatx2")
        net2 = tf.tensordot(flatx2, pc, 1, name="whiten2")
        net2 = tf.reshape(net2, np.shape(x_2), name="output2")
    return net1, net2


################################################################################
#    Title: One Hot Encoding
################################################################################
#    Description:
#        This function extends a matrix to one-hot encoding
#
#    Parameters:
#        y    Array of label values
#
#    Returns:
#        y_new    One hot encoded array of labels
################################################################################
def one_hot(y):
    n_values = np.max(y) + 1
    y_new = np.eye(n_values)[y[:, 0]]
    return y_new
