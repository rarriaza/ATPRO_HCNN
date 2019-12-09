import logging

import numpy as np
import os
import tensorflow as tf
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.datasets.cifar import load_batch
from tensorflow.python.keras.utils.data_utils import get_file

from .preprocess import load_preprocessed_data, build_fine2coarse_matrix
from .preprocess import preprocess_dataset_and_save

logger = logging.getLogger('CIFAR-100')


def get_cifar100(data_directory):
    (x, y_c), (x_test, y_test_c) = load_data('coarse', data_directory)
    (x, y), (x_test, y_test) = load_data('fine', data_directory)
    fine2coarse = build_fine2coarse_matrix(y_test, y_test_c)
    n_fine = len(np.unique(y_test))
    n_coarse = len(np.unique(y_test_c))
    if 'preprocessed_data' not in os.listdir(data_directory):
        logger.info("Preprocessing data")
        x, y, y_c, x_test, y_test, y_test_c = preprocess_dataset_and_save(
            x, y, y_c, x_test, y_test, y_test_c, data_directory, whitening=True)
    else:
        x, y, y_c, x_test, y_test, y_test_c = load_preprocessed_data(
            data_directory)

    logger.info("Casting data into float32")
    x = tf.cast(x, tf.float32)
    y = tf.cast(y, tf.float32)
    x_test = tf.cast(x_test, tf.float32)
    y_test = tf.cast(y_test, tf.float32)

    return (x, y), (x_test, y_test), fine2coarse, n_fine, n_coarse


def load_data(label_mode='fine', data_directory=None):
    """Loads CIFAR100 dataset. Reference: https://github.com/tensorflow/tensorflow/blob/v2.0.0/tensorflow/python/keras/datasets/cifar100.py
    Arguments:
        label_mode: one of "fine", "coarse".
    Returns:
        Tuple of Numpy arrays: `(x_train, y_train), (x_test, y_test)`.
    Raises:
        ValueError: in case of invalid `label_mode`.
    """
    if label_mode not in ['fine', 'coarse']:
        raise ValueError('`label_mode` must be one of `"fine"`, `"coarse"`.')

    dirname = 'cifar-100-python'
    origin = 'https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz'
    path = get_file(
        dirname,
        origin=origin,
        untar=True,
        file_hash='85cd44d02ba6437773c5bbd22e183051d648de2e7d6b014e1ef29b8'
                  '55ba677a7',
        cache_dir=data_directory)

    fpath = os.path.join(path, 'train')
    x_train, y_train = load_batch(fpath, label_key=label_mode + '_labels')

    fpath = os.path.join(path, 'test')
    x_test, y_test = load_batch(fpath, label_key=label_mode + '_labels')

    y_train = np.reshape(y_train, (len(y_train), 1))
    y_test = np.reshape(y_test, (len(y_test), 1))

    if K.image_data_format() == 'channels_last':
        x_train = x_train.transpose(0, 2, 3, 1)
        x_test = x_test.transpose(0, 2, 3, 1)

    return (x_train, y_train), (x_test, y_test)
