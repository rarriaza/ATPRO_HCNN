import tensorflow as tf
import logging

logger = logging.getLogger('ModelSaver')


class ModelSaver:
    def save_model(self, filename):
        logger.debug(f'Saving model to {filename}')
        tf.keras.models.save_model(filename)

    def load_model(self, filename):
        logger.debug(f'Loading model from {filename}')
        model = tf.keras.models.load_model(filename)
        return model
