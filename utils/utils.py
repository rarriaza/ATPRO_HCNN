import json
import logging

import numpy as np

logger = logging.getLogger("utils")


def get_error(y, yh):
    # Threshold
    yht = np.zeros(np.shape(yh))
    yht[np.arange(len(yh)), yh.argmax(1)] = 1
    # Evaluate Error
    error = np.count_nonzero(np.count_nonzero(y - yht, 1)) / len(y)
    return error


def freeze_model(model):
    for i in range(len(model.layers)):
        model.layers[i].trainable = False
    logger.info("Freezing parameters")


def unfreeze_model(model):
    for i in range(len(model.layers)):
        model.layers[i].trainable = False
    logger.info("Unfreezing parameters")


def freeze_layers(layers):
    for i in range(len(layers)):
        layers[i].trainable = False
    logger.info("Freezing parameters")


def unfreeze_layers(layers):
    for i in range(len(layers)):
        layers[i].trainable = True
    logger.info("Unfreezing parameters")


def write_results(results_file, results_dict):
    for a, b in results_dict.items():
        # Ensure that results_dict is made by numbers and lists only
        if type(b) is np.ndarray:
            results_dict[a] = b.tolist()
    json.dump(results_dict, open(results_file, 'w'))


def find_mismatch_error(pred_1, pred_2):
    n_pred = pred_1.shape[0]
    same = np.where(pred_1 == pred_2)[0]
    mis = (n_pred - same.shape[0]) / n_pred
    return mis
