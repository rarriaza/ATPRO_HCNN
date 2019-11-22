import logging

import os
import tensorflow as tf

logger = logging.getLogger('ModelSaver')


class ModelSaver:
    def save_model(self, filename, model):
        logger.debug(f'Saving model to {filename}')
        filepath = os.path.join(self.model_directory, "resnet_baseline.h5")
        if not os.path.exists(self.model_directory):
            os.makedirs(self.model_directory)
        model.save(filepath)
        # tf.keras.models.save_model(filename)

    def load_model(self, filename):
        logger.debug(f'Loading model from {filename}')
        model = tf.keras.models.load_model(filename)
        return model
