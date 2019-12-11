import json
import logging
from datetime import datetime

import numpy as np
import tensorflow as tf

import utils
from datasets.preprocess import shuffle_data

logger = logging.getLogger('HAT-CNN')


class VanillaCNN:
    def __init__(self, n_fine_categories, n_coarse_categories, input_shape,
                 logs_directory=None, model_directory=None, args=None):
        """
        Vanilla CNN
        """
        self.model_directory = model_directory
        self.args = args
        self.n_fine_categories = n_fine_categories
        self.n_coarse_categories = n_coarse_categories
        self.input_shape = input_shape

        self.full_model = None
        self.attention_units = 128

        current_time = datetime.now().strftime("%Y%m%d-%H%M%S")

        self.tbCallback_full = tf.keras.callbacks.TensorBoard(
            log_dir=logs_directory + '/' + current_time + '/full',
            update_freq='epoch')  # How often to write logs (default: once per epoch)


        self.training_params = {
            'batch_size': 64,
            'initial_epoch': 0,
            'lr': 1e-3,
            'step': 1,  # Save weights every this amount of epochs
            'stop': 10000,
            'patience': 5,
            'reduce_lr_after_patience_counts': 1,
            "validation_loss_threshold": 0,
            'lr_reduction_factor': 0.1
        }

        if self.args.debug_mode:
            self.training_params['step'] = 1
            self.training_params['stop'] = 1

        self.prediction_params = {
            'batch_size': 64
        }

    def save_best_full_model(self):
        logger.info(f"Saving best full model")
        loc = self.model_directory + "/vanilla_cnn_full_model.h5"
        self.full_model.save(loc)
        return loc

    def save_full_model(self):
        logger.info(f"Saving full model")
        loc = self.model_directory + "/vanilla_cnn_full_model_tmp.h5"
        self.full_model.save(loc)
        return loc

    def load_full_model(self, location):
        logger.info(f"Loading full model")
        self.full_model = tf.keras.models.load_model(location, custom_objects={'SGD': tf.keras.optimizers.SGD})

    def load_best_full_model(self):
        logger.info(f"Loading best full model")
        self.full_model = tf.keras.models.load_model(self.model_directory + "/vanilla_cnn_full_model.h5", custom_objects={'SGD': tf.keras.optimizers.SGD})

    def train(self, training_data, validation_data):
        x_train, y_train = training_data
        x_val, y_val = validation_data

        p = self.training_params
        val_thresh = p["validation_loss_threshold"]

        logger.info('Start Full Classification training')

        index = p['initial_epoch']

        tf.keras.backend.clear_session()
        self.full_model = self.build_model()
        loc = self.save_full_model()
        optim = tf.keras.optimizers.SGD(lr=p['lr'], nesterov=True, momentum=0.5)
        self.full_model.compile(optimizer=optim,
                                loss='categorical_crossentropy',
                                metrics=['accuracy'])

        tf.keras.backend.clear_session()


        prev_val_loss = float('inf')
        counts_patience = 0
        patience = p["patience"]
        while index < p['stop']:
            tf.keras.backend.clear_session()
            self.load_full_model(loc)
            self.full_model.compile(optimizer=optim,
                                    loss='categorical_crossentropy',
                                    metrics=['accuracy'])
            x_train, yc_train, _ = shuffle_data((x_train, y_train))
            full_fit = self.full_model.fit(x_train, yc_train,
                                           batch_size=p['batch_size'],
                                           initial_epoch=index,
                                           epochs=index + p["step"],
                                           validation_data=(x_val, y_val),
                                           callbacks=[self.tbCallback_full])
            val_loss = full_fit.history["val_loss"][-1]
            loc = self.save_full_model()
            if prev_val_loss - val_loss < val_thresh:
                counts_patience += 1
                logger.info(f"Counts to early stopping: {counts_patience}/{p['patience']}")
                if counts_patience >= patience:
                    break
                elif counts_patience % p["reduce_lr_after_patience_counts"] == 0:
                    new_val = optim.learning_rate * p["lr_reduction_factor"]
                    logger.info(f"LR is now: {new_val.numpy()}")
                    optim.learning_rate.assign(new_val)
                    self.load_best_full_model()
                    self.save_full_model()
            else:
                counts_patience = 0
                prev_val_loss = val_loss
                self.save_best_full_model()
            index += p["step"]

    def predict(self, testing_data, fine2coarse, results_file):
        x_test, y_test = testing_data
        yc_test = tf.linalg.matmul(y_test, fine2coarse)

        p = self.prediction_params

        self.load_best_full_model()
        self.build_model()

        ych_s = self.full_model.predict(x_test, batch_size=p['batch_size'])

        coarse_classification_error = utils.get_error(yc_test, ych_s)
        logger.info('Coarse Classifier Error: ' + str(coarse_classification_error))

        results_dict = {'Coarse Classifier Error': coarse_classification_error}

        self.write_results(results_file, results_dict=results_dict)

        # np.save(self.model_directory + "/fine_predictions.npy", yh_s)
        np.save(self.model_directory + "/coarse_predictions.npy", ych_s)
        # np.save(self.model_directory + "/fine_labels.npy", y_test)
        np.save(self.model_directory + "/coarse_labels.npy", yc_test)

        tf.keras.backend.clear_session()
        return ych_s

    def write_results(self, results_file, results_dict):
        for a, b in results_dict.items():
            # Ensure that results_dict is made by numbers and lists only
            if type(b) is np.ndarray:
                results_dict[a] = b.tolist()
        json.dump(results_dict, open(results_file, 'w'))

    def build_model(self, verbose=True):
        kernel_size = (3, 3)

        # CC Input
        inp = tf.keras.Input(shape=self.input_shape)

        # CC Model
        cc = tf.keras.layers.Conv2D(512, kernel_size, strides=(1, 1), padding='same')(inp)
        cc = tf.keras.layers.BatchNormalization()(cc)
        cc = tf.keras.layers.Activation("relu")(cc)
        cc = tf.keras.layers.Conv2D(256, kernel_size, strides=(2, 2), padding='same')(cc)
        cc = tf.keras.layers.BatchNormalization()(cc)
        cc = tf.keras.layers.Activation("relu")(cc)
        cc_att = tf.keras.layers.Conv2D(self.attention_units, kernel_size, strides=(2, 2), padding='same')(cc)
        cc_att = tf.keras.layers.BatchNormalization()(cc_att)
        cc_att = tf.keras.layers.Activation("relu", name='attention_layer')(cc_att)

        # FC Model
        fc = tf.keras.layers.Conv2D(128, kernel_size, strides=(1, 1), padding='same')(cc_att)
        fc = tf.keras.layers.BatchNormalization()(fc)
        fc = tf.keras.layers.Activation("relu")(fc)
        fc = tf.keras.layers.Conv2D(64, kernel_size, strides=(2, 2), padding='same')(fc)
        fc = tf.keras.layers.BatchNormalization()(fc)
        fc = tf.keras.layers.Activation("relu")(fc)
        fc = tf.keras.layers.Conv2D(32, kernel_size, strides=(2, 2), padding='same')(fc)
        fc = tf.keras.layers.BatchNormalization()(fc)
        fc = tf.keras.layers.Activation("relu")(fc)

        # FC Output
        fc = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(fc)
        fc_flat_out = tf.keras.layers.Flatten()(fc)
        fc_out = tf.keras.layers.Dense(256, activation='relu')(fc_flat_out)
        fc_out = tf.keras.layers.Dropout(0.3)(fc_out)
        fc_out = tf.keras.layers.Dense(self.n_fine_categories, activation='softmax')(fc_out)

        # Build FC
        model = tf.keras.models.Model(inputs=inp, outputs=fc_out)
        if verbose:
            print(model.summary())

        return model
