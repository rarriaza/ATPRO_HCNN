import json
import logging

import numpy as np
import os
import tensorflow as tf

import utils

logger = logging.getLogger('HDCNNBaseline')


class HDCNNBaseline:
    def __init__(self, n_fine_categories, n_coarse_categories,
                 logs_directory=None, model_directory=None, args=None):
        """
        HD-CNN baseline model

        """

        print(f"GPU is available: {tf.test.is_gpu_available()}")

        self.model_directory = model_directory
        self.args = args
        self.n_fine_categories = n_fine_categories
        self.n_coarse_categories = n_coarse_categories

        self.in_layer = self.define_input_layer()
        logger.debug(f"Creating full classifier with shared layers")
        self.full_classifier = self.build_full_classifier()
        logger.debug(f"Creating coarse classifier")
        self.coarse_classifier = self.build_coarse_classifier()
        self.fine_classifiers = {
            'models': [{} for i in range(n_coarse_categories)],
            'yhf': [{} for i in range(n_coarse_categories)]
        }
        for i in range(n_coarse_categories):
            logger.debug(f"Creating fine classifier {i}")
            model_i = self.build_fine_classifier()
            self.fine_classifiers['models'][i] = model_i

        self.tbCallBack = tf.keras.callbacks.TensorBoard(
            log_dir=logs_directory, histogram_freq=0,
            write_graph=True, write_images=True)

        self.shared_training_params = {
            'batch_size': 64,
            'initial_epoch': 0,
            'step': 5,  # Save weights every this amount of epochs
            'stop': 30
        }

        self.coarse_training_params = {
            'batch_size': 64,
            'step': 10,  # Save weights every this amount of epochs
            'coarse_stop': 40,
            'fine_stop': 50
        }

        self.fine_training_params = {
            'batch_size': 64,
            'step': 5,  # Save weights every this amount of epochs
            'coarse_stop': 5,
            'fine_stop': 10
        }

        self.prediction_params = {
            'batch_size': 64
        }

    def train_shared_layers(self, training_data, validation_data):
        logger.info('Training shared layers')
        x_train, y_train = training_data
        x_val, y_val = validation_data

        p = self.shared_training_params

        sgd_coarse = tf.keras.optimizers.SGD(
            lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        self.full_classifier.compile(optimizer=sgd_coarse,
                                     loss='categorical_crossentropy',
                                     metrics=['accuracy'])
        index = p['initial_epoch']
        while index < p['stop']:
            self.full_classifier.fit(x_train, y_train,
                                     batch_size=p['batch_size'],
                                     initial_epoch=index,
                                     epochs=index + p['step'],
                                     validation_data=(x_val, y_val),
                                     callbacks=[self.tbCallBack])
            index += p['step']
            self.save_model(os.path.join(self.model_directory,
                                         f"full_classifier_{index}"),
                            self.full_classifier)

    def train_coarse_classifier(self, training_data, validation_data,
                                fine2coarse):
        logger.info('Training coarse classifier')
        self.freeze_model(self.full_classifier)
        x_train, y_train = training_data
        x_val, y_val = validation_data

        logger.info("Transforming fine to coarse labels")
        y_train_c = np.dot(y_train, fine2coarse)
        y_val_c = np.dot(y_val, fine2coarse)

        p = self.coarse_training_params

        # Coarse training
        logger.info('Coarse training')
        sgd_coarse = tf.keras.optimizers.SGD(
            lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        self.coarse_classifier.compile(optimizer=sgd_coarse,
                                       loss='categorical_crossentropy',
                                       metrics=['accuracy'])

        index = self.shared_training_params['stop']
        while index < p['coarse_stop']:
            self.coarse_classifier.fit(x_train, y_train_c,
                                       batch_size=p['batch_size'],
                                       initial_epoch=index, epochs=index +
                                                                   p['step'],
                                       validation_data=(x_val,
                                                        y_val_c),
                                       callbacks=[self.tbCallBack])
            index += p['step']

        # Fine training
        sgd_fine = tf.keras.optimizers.SGD(
            lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
        self.coarse_classifier.compile(optimizer=sgd_fine,
                                       loss='categorical_crossentropy',
                                       metrics=['accuracy'])

        while index < p['fine_stop']:
            self.coarse_classifier.fit(x_train, y_train_c,
                                       batch_size=p['batch_size'],
                                       initial_epoch=index, epochs=index +
                                                                   p['step'],
                                       validation_data=(x_val,
                                                        y_val_c),
                                       callbacks=[self.tbCallBack])
            index += p['step']

    def train_fine_classifiers(self, training_data, validation_data,
                               fine2coarse):
        logger.info('Training fine classifiers')
        x_train, y_train = training_data
        x_val, y_val = validation_data

        p = self.fine_training_params

        for i in range(self.n_coarse_categories):
            logger.info(
                f'Training fine classifier {i + 1}/{self.n_coarse_categories}')
            # Get all training data for the coarse category
            ix = np.where([(y_train[:, j] == 1) for j in [
                k for k, e in enumerate(fine2coarse[:, i])
                if e != 0]])[1]
            x_tix = tf.gather(x_train, ix)
            y_tix = tf.gather(y_train, ix)

            # Get all validation data for the coarse category
            ix_v = np.where([(y_val[:, j] == 1) for j in [
                k for k, e in enumerate(fine2coarse[:, i])
                if e != 0]])[1]
            x_vix = tf.gather(x_val, ix_v)
            y_vix = tf.gather(y_val, ix_v)

            sgd_coarse = tf.keras.optimizers.SGD(
                lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
            self.fine_classifiers['models'][i].compile(
                optimizer=sgd_coarse, loss='categorical_crossentropy',
                metrics=['accuracy'])

            index = 0
            while index < p['coarse_stop']:
                self.fine_classifiers['models'][i].fit(
                    x_tix, y_tix, batch_size=p['batch_size'],
                    initial_epoch=index, epochs=index + p['step'],
                    validation_data=(x_vix, y_vix))
                index += p['step']

            sgd_fine = tf.keras.optimizers.SGD(
                lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
            self.fine_classifiers['models'][i].compile(
                optimizer=sgd_fine, loss='categorical_crossentropy',
                metrics=['accuracy'])

            while index < p['fine_stop']:
                self.fine_classifiers['models'][i].fit(
                    x_tix, y_tix, batch_size=p['batch_size'],
                    initial_epoch=index, epochs=index + p['step'],
                    validation_data=(x_vix, y_vix))
                index += p['step']

            yh_f = self.fine_classifiers['models'][i].predict(
                x_val[ix_v], batch_size=p['batch_size'])
            logger.info('Fine Classifier ' + str(i) + ' Error: ' +
                        str(utils.get_error(y_val[ix_v], yh_f)))

    def sync_parameters(self):
        """
        Synchronize parameters from full, coarse and all fine classifiers
        """
        logger.info("Copying parameters from full to coarse classifiers")
        for i in range(len(self.coarse_classifier.layers) - 1):
            self.coarse_classifier.layers[i].set_weights(
                self.full_classifier.layers[i].get_weights())

        logger.info("Copying parameters from full to all fine classifiers")
        for j, model_fine in enumerate(self.fine_classifiers['models']):
            logger.debug(
                f'Copying parameters from full to file classifier {j}')
            for i in range(len(model_fine.layers) - 1):
                model_fine.layers[i].set_weights(
                    self.full_classifier.layers[i].get_weights())

    def predict(self, testing_data, fine2coarse, results_file):
        logger.info("Predicting")
        x_test, y_test = testing_data

        p = self.prediction_params

        yh = np.zeros(np.shape(y_test))

        yh_s = self.full_classifier.predict(x_test, batch_size=p['batch_size'])

        single_classifier_error = utils.get_error(y_test, yh_s)
        logger.info('Single Classifier Error: ' + str(single_classifier_error))

        yh_c = self.coarse_classifier.predict(
            x_test, batch_size=p['batch_size'])
        y_c = np.dot(y_test, fine2coarse)

        coarse_classifier_error = utils.get_error(y_c, yh_c)
        logger.info('Coarse Classifier Error: ' + str(coarse_classifier_error))

        for i in range(self.n_coarse_categories):
            if i % 5 == 0:
                logger.info("Evaluating Fine Classifier: " + str(i))
            yh += np.multiply(yh_c[:, i].reshape((len(y_test)), 1),
                              self.fine_classifiers['yhf'][i])

        overall_error = utils.get_error(y_test, yh)
        logger.info('Overall Error: ' + str(overall_error))

        self.write_results(results_file, results={
            'Single Classifier Error': single_classifier_error,
            'Coarse Classifier Error': coarse_classifier_error,
            'Overall Error': overall_error
        })

        return yh

    def write_results(self, results_file, results_dict):
        for a, b in results_dict.items():
            # Ensure that results_dict is made by numbers and lists only
            if type(b) is np.ndarray:
                results_dict[a] = b.tolist()
        json.dump(results_dict, open(results_file, 'w'))

    def define_input_layer(self):
        in_layer = tf.keras.Input(
            shape=(32, 32, 3), dtype='float32', name='main_input')
        return in_layer

    def build_full_classifier(self):
        net = tf.keras.layers.Conv2D(384, 3, strides=1, padding='same',
                                     activation='elu')(self.in_layer)
        net = tf.keras.layers.MaxPooling2D((2, 2), padding='valid')(net)

        net = tf.keras.layers.Conv2D(
            384, 1, strides=1, padding='same', activation='elu')(net)
        net = tf.keras.layers.Conv2D(
            384, 2, strides=1, padding='same', activation='elu')(net)
        net = tf.keras.layers.Conv2D(
            640, 2, strides=1, padding='same', activation='elu')(net)
        net = tf.keras.layers.Conv2D(
            640, 2, strides=1, padding='same', activation='elu')(net)
        net = tf.keras.layers.Dropout(.2)(net)
        net = tf.keras.layers.MaxPooling2D((2, 2), padding='valid')(net)

        net = tf.keras.layers.Conv2D(
            640, 1, strides=1, padding='same', activation='elu')(net)
        net = tf.keras.layers.Conv2D(
            768, 2, strides=1, padding='same', activation='elu')(net)
        net = tf.keras.layers.Conv2D(
            768, 2, strides=1, padding='same', activation='elu')(net)
        net = tf.keras.layers.Conv2D(
            768, 2, strides=1, padding='same', activation='elu')(net)
        net = tf.keras.layers.Dropout(.3)(net)
        net = tf.keras.layers.MaxPooling2D((2, 2), padding='valid')(net)

        net = tf.keras.layers.Conv2D(
            768, 1, strides=1, padding='same', activation='elu')(net)
        net = tf.keras.layers.Conv2D(
            896, 2, strides=1, padding='same', activation='elu')(net)
        net = tf.keras.layers.Conv2D(
            896, 2, strides=1, padding='same', activation='elu')(net)
        net = tf.keras.layers.Dropout(.4)(net)
        net = tf.keras.layers.MaxPooling2D((2, 2), padding='valid')(net)

        net = tf.keras.layers.Conv2D(
            896, 3, strides=1, padding='same', activation='elu')(net)
        net = tf.keras.layers.Conv2D(
            1024, 2, strides=1, padding='same', activation='elu')(net)
        net = tf.keras.layers.Conv2D(
            1024, 2, strides=1, padding='same', activation='elu')(net)
        net = tf.keras.layers.Dropout(.5)(net)
        net = tf.keras.layers.MaxPooling2D((2, 2), padding='valid')(net)

        net = tf.keras.layers.Conv2D(
            1024, 1, strides=1, padding='same', activation='elu')(net)
        net = tf.keras.layers.Conv2D(
            1152, 2, strides=1, padding='same', activation='elu')(net)
        net = tf.keras.layers.Dropout(.6)(net)
        net = tf.keras.layers.MaxPooling2D((2, 2), padding='same')(net)

        net = tf.keras.layers.Flatten()(net)
        net = tf.keras.layers.Dense(1152, activation='elu')(net)
        net = tf.keras.layers.Dense(
            self.n_fine_categories, activation='softmax')(net)
        return tf.keras.models.Model(inputs=self.in_layer, outputs=net)

    def build_coarse_classifier(self):
        shared_layers = self.full_classifier.layers[-8].output
        net = tf.keras.layers.Conv2D(1024, 1, strides=1, padding='same',
                                     activation='elu')(shared_layers)
        net = tf.keras.layers.Conv2D(
            1152, 2, strides=1, padding='same', activation='elu')(net)
        net = tf.keras.layers.Dropout(.6)(net)
        net = tf.keras.layers.MaxPooling2D((2, 2), padding='same')(net)

        net = tf.keras.layers.Flatten()(net)
        net = tf.keras.layers.Dense(1152, activation='elu')(net)
        out_coarse = tf.keras.layers.Dense(
            self.n_coarse_categories, activation='softmax')(net)

        model_c = tf.keras.models.Model(
            inputs=self.in_layer, outputs=out_coarse)
        return model_c

    def build_fine_classifier(self):
        shared_layers = self.full_classifier.layers[-8].output
        net = tf.keras.layers.Conv2D(1024, 1, strides=1, padding='same',
                                     activation='elu')(shared_layers)
        net = tf.keras.layers.Conv2D(1152, 2, strides=1, padding='same',
                                     activation='elu')(net)
        net = tf.keras.layers.Dropout(.6)(net)
        net = tf.keras.layers.MaxPooling2D((2, 2), padding='same')(net)

        net = tf.keras.layers.Flatten()(net)
        net = tf.keras.layers.Dense(1152, activation='elu')(net)
        out_fine = tf.keras.layers.Dense(100, activation='softmax')(net)
        model_fine = tf.keras.models.Model(
            inputs=self.in_layer, outputs=out_fine)
        return model_fine

    def save_all_models(self, model_files_prefix):
        logger.info('Saving full classifier')
        self.save_model(model_files_prefix +
                        "_full_classifier.h5", self.full_classifier)
        logger.info('Saving coarse classifier')
        self.save_model(model_files_prefix +
                        "_coarse_classifier.h5", self.coarse_classifier)
        for i in range(self.n_coarse_categories):
            logger.info(f'Saving fine classifier {i}')
            self.save_model(model_files_prefix +
                            f"_fine_classifier_{i}.h5",
                            self.fine_classifiers["models"][i])

    def save_model(self, model_file, model):
        tf.keras.models.save_model(model, model_file)

    def load_model(self, model_file):
        return tf.keras.models.load_model(model_file)

    def load_models(self, model_files_prefix):
        logger.info('Loading full classifier')
        self.full_classifier = self.load_model(model_files_prefix +
                                               "_full_classifier.h5")
        logger.info('Loading coarse classifier')
        self.coarse_classifier = self.load_model(model_files_prefix +
                                                 "_coarse_classifier.h5")
        for i in range(self.n_coarse_categories):
            logger.info(f'Loading fine classifier {i}')
            self.fine_classifiers["models"][i] = self.load_model(
                model_files_prefix + f"_fine_classifier_{i}.h5")
