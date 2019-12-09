import logging
import os

import tensorflow as tf

logger = logging.getLogger('ModelSaver')


class ModelSaver:
    def save_model(self, filename, model):
        logger.debug(f'Saving model to {filename}')
        filepath = os.path.dirname(filename)
        if not os.path.exists(filepath):
            os.makedirs(filepath)
        tf.keras.models.save_model(model, filename)

    def load_model(self, filename):
        logger.debug(f'Loading model from {filename}')
        model = tf.keras.models.load_model(filename)
        return model
