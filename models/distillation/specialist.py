import os

import tensorflow as tf
import logging
import utils

import models.plugins as plugins

logger = logging.getLogger('Specialist')


class Specialist(plugins.ModelSaverPlugin):
    def __init__(self, input_shape, generalist_model_file, logs_directory=None, model_directory=None, args=None):
        """
        Specialist model based on ResNet-50

        """
        self.model_directory = model_directory
        self.args = args
        self.input_shape = input_shape

        logger.debug(f"Creating specialist classifier")
        self.full_classifier = self.build(generalist_model_file)

        self.tbCallBack = tf.keras.callbacks.TensorBoard(
            log_dir=logs_directory, histogram_freq=0,
            write_graph=True, write_images=True)

        self.training_params = {
            'batch_size': 64,
            'initial_epoch': 0,
            'step': 5,  # Save weights every this amount of epochs
            'stop': 30,
            'lr': 0.001,
            'lr_decay': 1e-6,
            'epochs': 5
        }

        self.prediction_params = {
            'batch_size': 64
        }

    def train(self, training_data, validation_data):
        x_train, y_train = training_data
        x_val, y_val = validation_data

        p = self.training_params

        adam = tf.keras.optimizers.Adam(lr=p['lr'], decay=p['lr_decay'])

        index = p['initial_epoch']

        # Freeze ResNet for tuning last layer
        # utils.freeze_layers(self.full_classifier.layers[:-2])

        self.full_classifier.compile(optimizer=adam,
                                     loss='categorical_crossentropy',
                                     metrics=['accuracy'])

        logger.info('Training specialist')
        while index < p["stop"]:
            self.full_classifier.fit(x_train, y_train,
                                     batch_size=p['batch_size'],
                                     initial_epoch=index,
                                     epochs=index + p['step'],
                                     validation_data=(x_val, y_val),
                                     callbacks=[self.tbCallBack])
            self.save_model(os.path.join(self.model_directory, "specialist.h5"))
            index += p['step']

        # Unfreeze ResNet for tuning last layer
        # utils.unfreeze_layers(self.full_classifier.layers[:-2])

    def predict(self, testing_data, results_file):
        x_test, y_test = testing_data

        p = self.prediction_params

        y_s = self.full_classifier.predict(x_test, batch_size=p['batch_size'])

        single_classifier_error = utils.get_error(y_test, y_s)
        logger.info('Specialist error: ' + str(single_classifier_error))

        results_dict = {'Specialist error': single_classifier_error}
        utils.write_results(results_file, results_dict=results_dict)

        return y_s

    def build(self, generalist_weights_file):
        model = tf.keras.models.load_model(generalist_weights_file)
        output = model.layers[-1]
        net = tf.keras.layers.Dense(2, activation='softmax')(output)
        return tf.keras.models.Model(inputs=model.input, outputs=net)
