import tensorflow as tf
import logging
import os
import numpy as np

logger = logging.getLogger('HDCNNBaseline')


class HDCNNBaseline:
    def __init__(self, n_fine_categories, n_coarse_categories, logs_directory=None, model_directory=None, args=None):
        """
        HD-CNN baseline model

        """
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
        self.training_params = {
            'batch_size': 64,
            'initial_epoch': 0,
            'step': 5,  # Save weights every this amount of epochs
            'epochs': 30
        }

        self.fine_tuning_params = {
            'batch_size': 64,
            'step': 10,  # Save weights every this amount of epochs
            'coarse_stop': 40,
            'fine_stop': 50
        }

    def train_shared_layers(self, training_data, validation_data):
        x_train, y_train = training_data
        x_val, y_val = validation_data

        p = self.training_params

        sgd_coarse = tf.keras.optimizers.SGD(
            lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        self.full_classifier.compile(optimizer=sgd_coarse,
                                     loss='categorical_crossentropy',
                                     metrics=['accuracy'])
        index = p['initial_epoch']
        while index < p['epochs']:
            self.full_classifier.fit(x_train, y_train,
                                     batch_size=p['batch_size'],
                                     initial_epoch=index,
                                     epochs=index + p['step'],
                                     validation_data=(x_val, y_val),
                                     callbacks=[self.tbCallBack])
            index += p['step']
            self.full_classifier.save_weights(
                os.path.join(self.model_directory, "fine_classifier",
                             str(index)))

    def sync_parameters(self):
        """
        Synchronize parameters from full, coarse and all fine classifiers
        """
        logger.info("Copying parameters from full to coarse classifiers")
        for i in range(len(self.coarse_classifier.layers)-1):
            self.coarse_classifier.layers[i].set_weights(
                self.full_classifier.layers[i].get_weights())

        logger.info("Copying parameters from full to all fine classifiers")
        for model_fine in self.fine_classifiers['models'].values():
            for i in range(len(model_fine.layers)-1):
                model_fine.layers[i].set_weights(
                    self.full_classifier.layers[i].get_weights())

    def train_coarse_classifier(self, training_data, validation_data,
                                fine2coarse):
        self.freeze_model(self.full_classifier)
        x_train, y_train = training_data
        x_val, y_val = validation_data

        logger.info("Transforming fine to coarse labels")
        y_train_c = np.dot(y_train, fine2coarse)
        y_val_c = np.dot(y_val, fine2coarse)

        p = self.fine_tuning_params

        # Coarse training
        sgd_coarse = tf.keras.optimizers.SGD(
            lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        self.coarse_classifier.compile(optimizer=sgd_coarse,
                                       loss='categorical_crossentropy',
                                       metrics=['accuracy'])

        index = self.training_params['stop']
        while index < p['coarse_stop']:
            self.coarse_classifier.fit(x_train, y_train_c,
                                       batch_size=p['batch_size'],
                                       index=index, epochs=index + p['step'],
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
                                       index=index, epochs=index + p['step'],
                                       validation_data=(x_val,
                                                        y_val_c),
                                       callbacks=[self.tbCallBack])
            index += p['step']

    def freeze_model(self, model):
        for i in range(len(model.layers)):
            model.layers[i].trainable = False

    def unfreeze_model(self, model):
        for i in range(len(model.layers)):
            model.layers[i].trainable = False

    def predict(self, testing_data, results_directory):
        pred = None
        return pred

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

    def load_weights(self, weights_file):
        self.full_classifier.load_weights(weights_file)
