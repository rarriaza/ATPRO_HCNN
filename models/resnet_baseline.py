import os

import tensorflow as tf
import logging
import numpy as np
import utils

import models.plugins as plugins

logger = logging.getLogger('ResNetBaseline')


class ResNetBaseline(plugins.ModelSaverPlugin):
    def __init__(self, n_fine_categories, n_coarse_categories, input_shape,
                 logs_directory=None, model_directory=None, args=None):
        """
        ResNet baseline model

        """
        self.model_directory = model_directory
        self.args = args
        self.n_fine_categories = n_fine_categories
        self.n_coarse_categories = n_coarse_categories
        self.input_shape = input_shape

        logger.debug(f"Creating full classifier with shared layers")
        self.full_classifier = self.build_full_classifier()

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
            'fine_tune_epochs': 5
        }

        self.prediction_params = {
            'batch_size': 64
        }

    def train(self, training_data, validation_data):
        x_train, y_train = training_data
        x_val, y_val = validation_data

        p = self.training_params

        adam_coarse = tf.keras.optimizers.Adam(lr=p['lr'], decay=p['lr_decay'])

        index = p['initial_epoch']

        # Freeze ResNet for tuning last layer
        utils.freeze_layers(self.full_classifier.layers[:-2])

        self.full_classifier.compile(optimizer=adam_coarse,
                                     loss='categorical_crossentropy',
                                     metrics=['accuracy'])

        logger.info('Fine tuning final layer')
        while index < p["fine_tune_epochs"]:
            self.full_classifier.fit(x_train, y_train,
                                     batch_size=p['batch_size'],
                                     initial_epoch=index,
                                     epochs=index + p['step'],
                                     validation_data=(x_val, y_val),
                                     callbacks=[self.tbCallBack])
            self.save_model(os.path.join(self.model_directory, "resnet_baseline.h5"))
            index += p['step']

        # Unfreeze ResNet for tuning last layer
        utils.unfreeze_layers(self.full_classifier.layers[:-2])

        # Recompile model
        adam_fine = tf.keras.optimizers.Adam(lr=0.001, decay=1e-6)
        self.full_classifier.compile(optimizer=adam_fine,
                                     loss='categorical_crossentropy',
                                     metrics=['accuracy'])

        # Main train
        while index < p['stop']:
            self.full_classifier.fit(x_train, y_train,
                                     batch_size=p['batch_size'],
                                     initial_epoch=index,
                                     epochs=index + p['step'],
                                     validation_data=(x_val, y_val),
                                     callbacks=[self.tbCallBack])
            self.save_model(os.path.join(self.model_directory, "resnet_baseline.h5"))
            index += p['step']

    def predict_fine(self, testing_data, results_file):
        x_test, y_test = testing_data

        p = self.prediction_params

        yh_s = self.full_classifier.predict(x_test, batch_size=p['batch_size'])

        single_classifier_error = utils.get_error(y_test, yh_s)
        logger.info('Single Classifier Error: '+str(single_classifier_error))

        results_dict = {'Single Classifier Error': single_classifier_error}
        utils.write_results(results_file, results_dict=results_dict)

        return yh_s

    def predict_coarse(self, testing_data, results_file, fine2coarse):
        x_test, y_test = testing_data

        p = self.prediction_params

        yh_s = self.full_classifier.predict(x_test, batch_size=p['batch_size'])

        single_classifier_error = utils.get_error(y_test, yh_s)
        logger.info('Single Classifier Error: ' + str(single_classifier_error))

        yh_c = np.dot(yh_s, fine2coarse)
        y_test_c = np.dot(y_test, fine2coarse)
        coarse_classifier_error = utils.get_error(y_test_c, yh_c)

        logger.info('Single Classifier Error: ' + str(coarse_classifier_error))
        results_dict = {'Single Classifier Error': single_classifier_error,
                        'Coarse Classifier Error': coarse_classifier_error}
        utils.write_results(results_file, results_dict=results_dict)

    def build_full_classifier(self):

        model = tf.keras.applications.resnet.ResNet50(include_top=False, weights='imagenet',
                                                      input_tensor=None, input_shape=self.input_shape,
                                                      pooling=None, classes=1000)
        net = tf.keras.layers.Flatten()(model.output)
        net = tf.keras.layers.Dense(
            self.n_fine_categories, activation='softmax')(net)
        return tf.keras.models.Model(inputs=model.input, outputs=net)
