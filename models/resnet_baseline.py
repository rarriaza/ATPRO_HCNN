import datetime
import logging

import tensorflow as tf
import numpy as np

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

        self.tbCallback = tf.keras.callbacks.TensorBoard(
            log_dir=self.logs_directory + '/' + current_time,
            update_freq='epoch')  # How often to write logs (default: once per epoch)

        self.training_params = {
            'batch_size': 64,
            'initial_epoch': 0,
            'step': 1,  # Save weights every this amount of epochs
            'stop': 1000,
            'lr': 1e-3,
            'val_thresh': 0,
            'patience': 10,
            'reduce_lr_after_patience_counts': 3,
            'lr_reduction_factor': 0.25
        }

        self.prediction_params = {
            'batch_size': 64
        }

    def train(self, training_data, validation_data):
        x_val, y_val = validation_data

        p = self.training_params

        optim = tf.keras.optimizers.SGD(lr=p['lr'])

        index = p['initial_epoch']

        prev_val_loss = float('inf')
        counts_patience = 0

        self.save_model(self.model_directory + "/vanilla_tmp.h5", self.full_classifier)

        while index < p['stop']:
            tf.keras.backend.clear_session()
            self.full_classifier = self.load_model(self.model_directory + "/vanilla_tmp.h5")

            self.full_classifier.compile(optimizer=optim,
                                         loss='categorical_crossentropy',
                                         metrics=['accuracy'])

            # logger.info('Training coarse stage')
            x_train, y_train, _ = shuffle_data(training_data)
            fc = self.full_classifier.fit(x_train, y_train,
                                          batch_size=p['batch_size'],
                                          initial_epoch=index,
                                          epochs=index + p['step'],
                                          validation_data=(x_val, y_val),
                                          callbacks=[self.tbCallback])
            val_loss = fc.history['val_loss'][0]

            self.save_model(self.model_directory + "/vanilla_tmp.h5", self.full_classifier)

            if prev_val_loss - val_loss < p['val_thresh']:
                counts_patience += 1
                logger.info(f"Counts to early stopping: {counts_patience}/{p['patience']}")
                if counts_patience >= p['patience']:
                    break
                elif counts_patience % p["reduce_lr_after_patience_counts"] == 0:
                    new_val = optim.learning_rate * p["lr_reduction_factor"]
                    logger.info(f"LR is now: {new_val.numpy()}")
                    optim.learning_rate.assign(new_val)
            else:
                counts_patience = 0
                prev_val_loss = val_loss
                self.save_model(self.model_directory + "/vanilla.h5", self.full_classifier)

            index += p['step']

    def predict_fine(self, testing_data, results_file, fine2coarse):
        x_test, y_test = testing_data

        self.full_classifier = self.load_model(self.model_directory + '/vanilla.h5')

        p = self.prediction_params

        yh_s = self.full_classifier.predict(x_test, batch_size=p['batch_size'])

        single_classifier_error = utils.get_error(y_test, yh_s)
        logger.info('Single Classifier Error: ' + str(single_classifier_error))

        results_dict = {'Single Classifier Error': single_classifier_error}
        utils.write_results(results_file, results_dict=results_dict)

        np.save(self.model_directory + "/fine_predictions.npy", yh_s)
        # np.save(self.model_directory + "/coarse_predictions.npy", ych_s)
        np.save(self.model_directory + "/fine_labels.npy", y_test)
        # np.save(self.model_directory + "/coarse_labels.npy", yc_test)

        return yh_s

    def build_full_classifier(self):
        model = tf.keras.applications.resnet.ResNet50(include_top=False, weights='imagenet',
                                                      input_tensor=None, input_shape=self.input_shape,
                                                      pooling=None, classes=1000)
        inp = tf.keras.Input(shape=model.input.shape[1:])
        net = model(inp)
        net = tf.keras.layers.Flatten()(net)
        net = tf.keras.layers.Dense(self.n_fine_categories, activation='softmax')(net)
        return tf.keras.models.Model(inputs=inp, outputs=net)
