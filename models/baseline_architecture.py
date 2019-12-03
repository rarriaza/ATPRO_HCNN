import datetime
import json
import logging

import numpy as np
import tensorflow as tf
from random import randint

import utils
from datasets.preprocess import shuffle_data
from models.resnet_common import ResNet50

logger = logging.getLogger('BaselineArchitecture')


class BaselineArchitecture:
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
        self.cc, self.fc = None, None

        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

        self.tbCallback_coarse = tf.keras.callbacks.TensorBoard(
            log_dir=logs_directory + '/' + current_time + '/coarse',
            update_freq='epoch')  # How often to write logs (default: once per epoch)
        self.tbCallback_fine = tf.keras.callbacks.TensorBoard(
            log_dir=logs_directory + '/' + current_time + '/fine',
            update_freq='epoch')  # How often to write logs (default: once per epoch)

        self.training_params = {
            'batch_size': 64,
            'initial_epoch': 0,
            'lr_coarse': 3e-6,
            'lr_fine': 3e-6,
            'step': 1,  # Save weights every this amount of epochs
            'stop': 10000,
            'patience': 10,
            'patience_decrement': 10,
            'decrement_lr': 0.2
        }

        self.prediction_params = {
            'batch_size': 64
        }

    def save_best_cc_model(self):
        logger.info(f"Saving best cc model")
        loc = self.model_directory + "/baseline_arch_cc.h5"
        self.cc.save(loc)
        return loc

    def save_best_fc_model(self):
        logger.info(f"Saving best fc model")
        loc = self.model_directory + "/baseline_arch_fc.h5"
        self.fc.save(loc)
        return loc

    def save_best_cc_both_model(self):
        logger.info(f"Saving best cc both model")
        loc = self.model_directory + "/baseline_arch_cc_both.h5"
        self.cc.save(loc)
        return loc

    def save_best_fc_both_model(self):
        logger.info(f"Saving best fc both model")
        loc = self.model_directory + "/baseline_arch_fc_both.h5"
        self.fc.save(loc)
        return loc

    def save_cc_model(self, epochs, val_accuracy):
        logger.info(f"Saving cc model")
        loc = self.model_directory + f"/baseline_arch_cc_epochs_{epochs:02d}_valacc_{val_accuracy:.4}.h5"
        self.cc.save(loc)
        return loc

    def save_fc_model(self, epochs, val_accuracy):
        logger.info(f"Saving fc model")
        loc = self.model_directory + f"/baseline_arch_fc_epochs_{epochs:02d}_valacc_{val_accuracy:.4}.h5"
        self.fc.save(loc)
        return loc

    def save_cc_both_model(self, epochs, val_accuracy):
        logger.info(f"Saving cc model")
        loc = self.model_directory + f"/baseline_arch_cc_both_epochs_{epochs:02d}_valacc_{val_accuracy:.4}.h5"
        self.cc.save(loc)
        return loc

    def save_fc_both_model(self, epochs, val_accuracy):
        logger.info(f"Saving fc model")
        loc = self.model_directory + f"/baseline_arch_fc_both_epochs_{epochs:02d}_valacc_{val_accuracy:.4}.h5"
        self.fc.save(loc)
        return loc

    def load_best_cc_model(self):
        logger.info(f"Loading best cc model")
        self.load_cc_model(self.model_directory + "/baseline_arch_cc.h5")

    def load_best_fc_model(self):
        logger.info(f"Loading best fc model")
        self.load_fc_model(self.model_directory + "/baseline_arch_fc.h5")

    def load_cc_model(self, location):
        logger.info(f"Loading cc model")
        self.cc = tf.keras.models.load_model(location)

    def load_fc_model(self, location):
        logger.info(f"Loading fc model")
        self.fc = tf.keras.models.load_model(location)

    def train_coarse(self, training_data, validation_data, fine2coarse):
        x_train, y_train = training_data
        yc_train = tf.linalg.matmul(y_train, fine2coarse)

        x_val, y_val = validation_data
        yc_val = tf.linalg.matmul(y_val, fine2coarse)

        del y_train, y_val

        p = self.training_params

        logger.debug(f"Creating coarse classifier with shared layers")
        self.cc, _ = self.build_cc_fc()
        adam_coarse = tf.keras.optimizers.Adam(lr=p['lr_coarse'])
        self.cc.compile(optimizer=adam_coarse,
                        loss='categorical_crossentropy',
                        metrics=['accuracy'])

        loc = self.save_cc_model(0, 0.0)

        logger.info('Start Coarse Classification Training')

        index = p['initial_epoch']

        best_model = loc
        prev_val_loss = 0.0
        val_loss = 0
        counts_patience = 0
        patience = p["patience"]
        decremented = 0
        while index < p['stop']:
            tf.keras.backend.clear_session()
            self.load_cc_model(loc)
            x_train, yc_train, _ = shuffle_data((x_train, yc_train))
            cc_fit = self.cc.fit(x_train, yc_train,
                                 batch_size=p['batch_size'],
                                 initial_epoch=index,
                                 epochs=index + p["step"],
                                 validation_data=(x_val, yc_val),
                                 callbacks=[self.tbCallback_coarse])
            val_loss = cc_fit.history["val_loss"][-1]
            val_acc = cc_fit.history["val_accuracy"][-1]
            loc = self.save_cc_model(index, val_acc)
            if val_loss - prev_val_loss < 0:
                if counts_patience == 0:
                    best_model = loc
                counts_patience += 1
                logger.info(f"Counts to early stopping: {counts_patience}/{p['patience']}")
                if counts_patience >= patience:
                    break
            else:
                counts_patience = 0
                prev_val_loss = val_loss
            index += p["step"]
        if best_model is not None:
            tf.keras.backend.clear_session()
            self.load_cc_model(best_model)

        best_model = self.save_best_cc_model()
        # best_model = loc  # This is just for debugging purposes
        return best_model

    def train_fine(self, training_data, validation_data, fine2coarse):
        x_train, y_train = training_data
        yc_train = tf.linalg.matmul(y_train, fine2coarse)
        x_val, y_val = validation_data
        yc_val = tf.linalg.matmul(y_val, fine2coarse)

        p = self.training_params

        feature_map = self.get_feature_input_for_fc(x_train)
        feature_map_val = self.get_feature_input_for_fc(x_val)

        _, self.fc = self.build_cc_fc(verbose=False)
        adam_fine = tf.keras.optimizers.Adam(lr=p['lr_fine'])
        self.fc.compile(optimizer=adam_fine,
                        loss='categorical_crossentropy',
                        metrics=['accuracy'])
        loc = self.save_fc_model(0, 0.0, 1e-5)
        tf.keras.backend.clear_session()
        # self.load_fc_model(loc)

        logger.info('Start Fine Classification Training')

        index = p['initial_epoch']

        prev_val_loss = 0.0
        val_loss = 0
        counts_patience = 0
        patience = p["patience"]
        decremented = 0
        best_model = loc
        while index < p['stop']:
            tf.keras.backend.clear_session()
            self.load_fc_model(loc)
            s = randint(0, 10000)
            feature_map, y_train, inds = shuffle_data((feature_map, y_train), random_state=s)
            yc_train = tf.gather(yc_train, inds)

            fc_fit = self.fc.fit([feature_map, yc_train], y_train,
                                 batch_size=p['batch_size'],
                                 initial_epoch=index,
                                 epochs=index + p["step"],
                                 validation_data=([feature_map_val, yc_val], y_val),
                                 callbacks=[self.tbCallback_fine])
            val_loss = fc_fit.history["val_loss"][-1]
            val_acc = fc_fit.history["val_accuracy"][-1]
            loc = self.save_fc_model(index, val_acc)
            if val_loss - prev_val_loss < 5e-3:
                if counts_patience == 0:
                    best_model = loc
                counts_patience += 1
                logger.info(f"Counts to early stopping: {counts_patience}/{p['patience']}")
                if counts_patience >= patience:
                    break
            else:
                counts_patience = 0
                prev_val_loss = val_loss
            if decremented >= p['patience_decrement']:
                break
            index += p["step"]
        if best_model is not None:
            tf.keras.backend.clear_session()
            self.load_fc_model(best_model)

        best_model = self.save_best_fc_model()
        # best_model = loc  This is just for debugging purposes
        return best_model

    def train_both(self, training_data, validation_data, fine2coarse):
        x_train, y_train = training_data
        x_val, y_val = validation_data
        yc_train = tf.linalg.matmul(y_train, fine2coarse)
        yc_val = tf.linalg.matmul(y_val, fine2coarse)

        p = self.training_params

        logger.info('Start Full Classification training')

        index = p['initial_epoch']

        tf.keras.backend.clear_session()
        loc_cc = "./saved_models/attention/baseline_arch_cc.h5"
        loc_fc = "./saved_models/attention/baseline_arch_fc.h5"
        self.load_cc_model(loc_cc)
        self.load_fc_model(loc_fc)

        self.build_full_model()
        adam_fine = tf.keras.optimizers.Adam(lr=p['lr_fine'])
        self.full_model.compile(optimizer=adam_fine,
                                loss='categorical_crossentropy',
                                metrics=['accuracy'])

        loc_cc = self.save_cc_model(0, 0.0)
        loc_fc = self.save_fc_model(0, 0.0)
        best_model_cc = loc_cc
        best_model_fc = loc_fc
        tf.keras.backend.clear_session()
        self.load_fc_model(loc_fc)
        self.load_cc_model(loc_cc)

        prev_val_loss_fine = 0.0
        counts_patience = 0
        patience = p["patience"]
        while index < p['stop']:
            tf.keras.backend.clear_session()
            self.load_cc_model(loc_cc)
            self.load_fc_model(loc_fc)
            self.build_full_model()
            adam_fine = tf.keras.optimizers.Adam(lr=p['lr_fine'])
            self.full_model.compile(optimizer=adam_fine,
                                    loss='categorical_crossentropy',
                                    metrics=['accuracy'])
            x_train, y_train, inds = shuffle_data((x_train, y_train))
            yc_train = tf.gather(yc_train, inds)
            full_fit = self.full_model.fit(x_train, [y_train, yc_train],
                                           batch_size=p['batch_size'],
                                           initial_epoch=index,
                                           epochs=index + p["step"],
                                           validation_data=(x_val, [y_val, yc_val]),
                                           callbacks=[self.tbCallback_coarse])
            val_loss_fine = full_fit.history["val_model_1_loss"][-1]
            val_loss_coarse = full_fit.history["val_dense_loss"][-1]
            val_acc_fine = full_fit.history["val_model_1_accuracy"][-1]
            val_acc_coarse = full_fit.history["val_dense_accuracy"][-1]
            loc_cc = self.save_cc_both_model(index, val_acc_coarse)
            loc_fc = self.save_fc_both_model(index, val_acc_fine)
            if val_loss_fine - prev_val_loss_fine < 0:
                if counts_patience == 0:
                    best_model_cc = loc_cc
                    best_model_fc = loc_fc
                counts_patience += 1
                logger.info(f"Counts to early stopping: {counts_patience}/{p['patience']}")
                if counts_patience >= patience:
                    break
            else:
                counts_patience = 0
                prev_val_loss_fine = val_loss_fine
            index += p["step"]
        if best_model_cc is not None and best_model_fc is not None:
            tf.keras.backend.clear_session()
            self.load_cc_model(best_model_cc)
            self.load_fc_model(best_model_fc)

        # best_model = self.save_best_full_model()
        best_model_cc = self.save_best_cc_both_model()
        best_model_fc = self.save_best_fc_both_model()
        # best_model = loc  This is just for debugging purposes
        return best_model_cc, best_model_fc

    def predict_coarse(self, testing_data, fine2coarse, results_file):
        x_test, y_test = testing_data
        yc_test = tf.linalg.matmul(y_test, fine2coarse)

        p = self.prediction_params

        yc_pred = self.cc.predict(x_test, batch_size=p['batch_size'])

        coarse_classifier_error = utils.get_error(yc_test, yc_pred)

        logger.info('Coarse Classifier Error: ' + str(coarse_classifier_error))
        results_dict = {'Coarse Classifier Error': coarse_classifier_error}
        self.write_results(results_file, results_dict=results_dict)

        tf.keras.backend.clear_session()
        return yc_pred

    def predict_fine(self, testing_data, results_file):
        x_test_feat, yc_pred, y_test = testing_data

        p = self.prediction_params

        yh_s = self.fc.predict([x_test_feat, yc_pred], batch_size=p['batch_size'])

        single_classifier_error = utils.get_error(y_test, yh_s)
        logger.info('Single Classifier Error: ' + str(single_classifier_error))

        results_dict = {'Single Classifier Error': single_classifier_error}
        self.write_results(results_file, results_dict=results_dict)

        tf.keras.backend.clear_session()
        return yh_s

    def write_results(self, results_file, results_dict):
        for a, b in results_dict.items():
            # Ensure that results_dict is made by numbers and lists only
            if type(b) is np.ndarray:
                results_dict[a] = b.tolist()
        json.dump(results_dict, open(results_file, 'w'))

    def build_cc_fc(self, verbose=True):
        model_1, model_2 = ResNet50(include_top=False, weights='imagenet',
                                    input_tensor=None, input_shape=self.input_shape,
                                    pooling=None, classes=1000)

        # Define CC Prediction Block
        cc_flat = tf.keras.layers.Flatten()(model_1.output)
        cc_out = tf.keras.layers.Dense(
            self.n_coarse_categories, activation='softmax')(cc_flat)

        cc_model = tf.keras.models.Model(inputs=model_1.input, outputs=cc_out)
        if verbose:
            print(cc_model.summary())

        # fine classification
        fc_flat = tf.keras.layers.Flatten()(model_2.output)
        # Define as Input the prediction of coarse labels
        fc_in_cc_labels = tf.keras.layers.Input(shape=self.n_coarse_categories)
        # Add the CC prediction to the flatten layer just before the output layer
        fc_flat_cc = tf.keras.layers.concatenate([fc_flat, fc_in_cc_labels])
        fc_out = tf.keras.layers.Dense(
            self.n_fine_categories, activation='softmax')(fc_flat_cc)

        fc_model = tf.keras.models.Model(inputs=[model_2.input, fc_in_cc_labels], outputs=fc_out)
        if verbose:
            print(fc_model.summary())

        return cc_model, fc_model

    def get_feature_input_for_fc(self, data):
        batch_size = self.prediction_params['batch_size']
        self.cc, _ = self.build_cc_fc(verbose=False)

        self.load_best_cc_model()

        feature_model = tf.keras.models.Model(inputs=self.cc.input,
                                              outputs=self.cc.get_layer('conv2_block3_out').output)
        feature_map = feature_model.predict(data, batch_size=batch_size)

        tf.keras.backend.clear_session()
        return feature_map

    def build_full_model(self):
        cc_mod_feat = tf.keras.Model(self.cc.input, [self.cc.layers[-3].output, self.cc.output])
        cc_mod_feat._name = "dont_care"
        cc_feat = cc_mod_feat(self.cc.input)
        fc_out = self.fc([cc_feat[0], self.cc.output])
        self.full_model = tf.keras.Model(inputs=self.cc.inputs, outputs=[fc_out, self.cc.output])
