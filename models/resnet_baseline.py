import datetime
import logging

import tensorflow as tf

import models.plugins as plugins
import utils
from datasets.preprocess import shuffle_data

logger = logging.getLogger('ResNetBaseline')


class ResNetBaseline(plugins.ModelSaverPlugin):
    def __init__(self, n_fine_categories, n_coarse_categories, input_shape,
                 logs_directory, model_directory=None, args=None):
        """
        ResNet baseline model

        """
        self.model_directory = model_directory
        self.args = args
        self.n_fine_categories = n_fine_categories
        self.n_coarse_categories = n_coarse_categories
        self.input_shape = input_shape
        self.loss_fun = None
        self.adam_coarse = None
        self.adam_fine = None

        logger.debug(f"Creating full classifier with shared layers")
        self.full_classifier = self.build_full_classifier()

        self.logs_directory = logs_directory

        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

        self.tbCallback_train = tf.keras.callbacks.TensorBoard(
            log_dir=self.logs_directory + '/' + current_time,
            update_freq='epoch')  # How often to write logs (default: once per epoch)
        self.early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
        self.reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                                                              patience=5, min_lModelCheckpointr=0.0000001)
        self.model_checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=model_directory + "resnet_baseline_{epoch:02d}-epochs.h5",
                                                                   save_freq=90000)

        self.training_params = {
            'batch_size': 64,
            'initial_epoch': 0,
            'step': 5,  # Save weights every this amount of epochs
            'stop': 100,
            'lr': 0.00003,
        }

        self.prediction_params = {
            'batch_size': 64
        }

    def train(self, training_data, validation_data):
        x_val, y_val = validation_data

        p = self.training_params

        self.adam_coarse = tf.keras.optimizers.Adam(lr=p['lr'])
        self.loss_fun = tf.keras.losses.CategoricalCrossentropy()

        index = p['initial_epoch']

        self.full_classifier.compile(optimizer=self.adam_coarse,
                                     loss='categorical_crossentropy',
                                     metrics=['accuracy'])

        # logger.info('Training coarse stage')
        training_data = shuffle_data(training_data)
        x_train, y_train = training_data
        self.full_classifier.fit(x_train, y_train,
                                 batch_size=p['batch_size'],
                                 initial_epoch=index,
                                 epochs=index + p['stop'],
                                 validation_data=(x_val, y_val),
                                 callbacks=[self.tbCallback_train, self.early_stopping,
                                            self.reduce_lr, self.model_checkpoint])

    def predict_fine(self, testing_data, results_file):
        x_test, y_test = testing_data

        p = self.prediction_params

        yh_s = self.full_classifier.predict(x_test, batch_size=p['batch_size'])

        single_classifier_error = utils.get_error(y_test, yh_s)
        logger.info('Single Classifier Error: ' + str(single_classifier_error))

        results_dict = {'Single Classifier Error': single_classifier_error}
        utils.write_results(results_file, results_dict=results_dict)

        return yh_s

    def build_full_classifier(self):

        model = tf.keras.applications.resnet.ResNet50(include_top=False, weights='imagenet',
                                                      input_tensor=None, input_shape=self.input_shape,
                                                      pooling=None, classes=1000)
        net = tf.keras.layers.Flatten()(model.output)
        net = tf.keras.layers.Dense(
            self.n_fine_categories, activation='softmax')(net)
        return tf.keras.models.Model(inputs=model.input, outputs=net)

