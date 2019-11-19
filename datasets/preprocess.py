import tensorflow as tf
import time
import logging
import numpy as np
import os

logger = logging.getLogger('preprocess')


def shuffle_data(data, random_state=0):
    X, y = data
    n = len(X)
    inds = tf.range(n)
    inds = tf.random.shuffle(inds, seed=random_state)
    X = tf.gather(X, inds)
    y = tf.gather(y, inds)
    return X, y


def train_test_split(data, test_size=.1):
    X, y = data
    n = len(X)
    n_test = int(round(test_size*n))

    inds = tf.range(n)

    inds_val = inds[:n_test]
    inds_train = inds[n_test:]

    X_train = tf.gather(X, inds_train)
    y_train = tf.gather(y, inds_train)
    X_val = tf.gather(X, inds_val)
    y_val = tf.gather(y, inds_val)

    return (X_train, y_train), (X_val, y_val)


def preprocess_dataset(x, y, y_c, x_test, y_test, y_test_c, whitening=False):
    # One-hot
    logger.debug(f'One hot: shape of y before: {y.shape}')
    y = one_hot(y)
    logger.debug(f'One hot: shape of y after: {y.shape}')
    logger.debug(f'One hot: shape of y_test before: {y_test.shape}')
    y_test = one_hot(y_test)
    logger.debug(f'One hot: shape of y_test after: {y_test.shape}')

    logger.debug(f'One hot: shape of y_c before: {y_c.shape}')
    y_c = one_hot(y_c)
    logger.debug(f'One hot: shape of y_c after: {y_c.shape}')
    logger.debug(f'One hot: shape of y_test_c before: {y_test_c.shape}')
    y_test_c = one_hot(y_test_c)
    logger.debug(f'One hot: shape of y_test_c after: {y_test_c.shape}')

    # ZCA whitening
    if whitening:
        logger.info("ZCA whitening")
        time1 = time.time()
        x, x_test = zca(x, x_test)
        time2 = time.time()
        logger.info(f'Time Elapsed - ZCA Whitening: {time2 - time1}')

    # Augmentation
    logger.info(
        "Pad images by 4 pixels, randomly crop them and then randomly flip them"
    )
    time1 = time.time()
    x, y, y_c = per_img_preprocess(x, y, y_c)
    time2 = time.time()
    logger.info(f'Time Elapsed - Image augmentation: {time2 - time1}')
    return x, y, y_c, x_test, y_test, y_test_c


def preprocess_dataset_and_save(x, y, y_c, x_test, y_test, y_test_c,
                                data_directory):
    x, y, y_c, x_test, y_test, y_test_c = preprocess_dataset(x, y, y_c, x_test, y_test, y_test_c)
    x_np = np.array(x)
    y_np = np.array(y)
    y_c_np = np.array(y_c)
    x_test_np = np.array(x_test)
    y_test_np = np.array(y_test)
    y_test_c_np = np.array(y_test_c)
    os.makedirs(data_directory + '/preprocessed_data', exist_ok=True)
    np.save(data_directory + '/preprocessed_data/x', x_np)
    np.save(data_directory + '/preprocessed_data/x_test', x_test_np)
    np.save(data_directory + '/preprocessed_data/y', y_np)
    np.save(data_directory + '/preprocessed_data/y_test', y_test_np)
    np.save(data_directory + '/preprocessed_data/y_c', y_c_np)
    np.save(data_directory + '/preprocessed_data/y_test_c', y_test_c_np)
    return x, y, y_c, x_test, y_test, y_test_c


def load_preprocessed_data(data_directory):
    x = np.load(data_directory + '/preprocessed_data/x.npy')
    x = tf.convert_to_tensor(x)
    x_test = np.load(data_directory + '/preprocessed_data/x_test.npy')
    x_test = tf.convert_to_tensor(x_test)
    y = np.load(data_directory + '/preprocessed_data/y.npy')
    y = tf.convert_to_tensor(y)
    y_test = np.load(data_directory + '/preprocessed_data/y_test.npy')
    y_test = tf.convert_to_tensor(y_test)
    y_c = np.load(data_directory + '/preprocessed_data/y_c.npy')
    y_c = tf.convert_to_tensor(y_c)
    y_test_c = np.load(data_directory + '/preprocessed_data/y_test_c.npy')
    y_test_c = tf.convert_to_tensor(y_test_c)
    return x, y, y_c, x_test, y_test, y_test_c


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

################################################################################
#    Title: Per img preprocess
################################################################################
#    Description:
#        This function pads images by 4 pixels, randomly crops them, then
#        randomly flips them
#
#    Parameters:
#        x_1           Array of MxNxC images to compute the ZCA Whitening
#        x_2           Array of MxNxC images to apply the ZCA transform
#        num_batch    Number of batches to do the computation
#
#    Returns:
#        An array of MxNxC zca whitened images
################################################################################


@tf.function
def per_img_preprocess(X, y, y_c):
    with tf.name_scope('Preproc'):
        net = tf.map_fn(lambda img: tf.image.flip_left_right(img), X)
        net = tf.map_fn(lambda img: tf.image.rot90(img), net)
        net = tf.image.resize_with_crop_or_pad(net, 40, 40)
        net = tf.map_fn(lambda img: tf.image.random_crop(
            img, [32, 32, 3]), net)
        net1 = tf.image.resize_with_crop_or_pad(X, 40, 40)
        net1 = tf.map_fn(lambda img: tf.image.random_crop(
            img, [32, 32, 3]), net1)
        net = tf.concat([net, net1], 0)
        net = tf.random.shuffle(net, seed=0)
        net_labels = tf.concat([y, y], 0)
        net_labels = tf.random.shuffle(net_labels, seed=0)
        net_labels_c = tf.concat([y_c, y_c], 0)
        net_labels_c = tf.random.shuffle(net_labels_c, seed=0)
        net = tf.map_fn(lambda img: tf.image.random_flip_up_down(img), net)
    return net, net_labels, net_labels_c


def build_fine2coarse_matrix(y, y_c):
    fine_categories = len(np.unique(y))
    coarse_categories = len(np.unique(y_c))
    fine2coarse = np.zeros((fine_categories, coarse_categories))
    for i in range(coarse_categories):
        index = np.where(y_c[:, 0] == i)[0]
        fine_cat = np.unique([y[j, 0] for j in index])
        for j in fine_cat:
            fine2coarse[j, i] = 1
    return fine2coarse
