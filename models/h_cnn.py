import json
import logging
from datetime import datetime

import numpy as np
import tensorflow as tf

import utils
from datasets.preprocess import shuffle_data
from models.include.attention_layer import SelfAttention
from models.hat_resnet import NormL

logger = logging.getLogger('H-CNN')


class HCNN:
    def __init__(self, n_fine_categories, n_coarse_categories, input_shape,
                 logs_directory=None, model_directory=None, args=None):
        """
        H CNN
        """
        self.model_directory = model_directory
        self.args = args
        self.n_fine_categories = n_fine_categories
        self.n_coarse_categories = n_coarse_categories
        self.input_shape = input_shape

        self.cc, self.fc, self.full_model = None, None, None
        self.attention = None
        self.attention_units = 128

        current_time = datetime.now().strftime("%Y%m%d-%H%M%S")

        self.tbCallback_coarse = tf.keras.callbacks.TensorBoard(
            log_dir=logs_directory + '/' + current_time + '/coarse',
            update_freq='epoch')  # How often to write logs (default: once per epoch)
        self.tbCallback_fine = tf.keras.callbacks.TensorBoard(
            log_dir=logs_directory + '/' + current_time + '/fine',
            update_freq='epoch')  # How often to write logs (default: once per epoch)
        self.tbCallback_full = tf.keras.callbacks.TensorBoard(
            log_dir=logs_directory + '/' + current_time + '/full',
            update_freq='epoch')  # How often to write logs (default: once per epoch)


        self.training_params = {
            'batch_size': 64,
            'initial_epoch': 0,
            'lr_coarse': 1e-3,
            'lr_fine': 1e-3,
            'lr_full': 1e-5,
            'step': 1,  # Save weights every this amount of epochs
            'step_full': 1,
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

    def save_best_cc_model(self):
        logger.info(f"Saving best cc model")
        loc = self.model_directory + "/hcnn_cc.h5"
        self.cc.save(loc)
        return loc

    def save_best_fc_model(self):
        logger.info(f"Saving best fc model")
        loc = self.model_directory + "/hcnn_fc.h5"
        self.fc.save(loc)
        return loc

    def save_best_cc_both_model(self):
        logger.info(f"Saving best cc both model")
        loc = self.model_directory + "/hcnn_cc_both.h5"
        self.cc.save(loc)
        return loc

    def save_best_fc_both_model(self):
        logger.info(f"Saving best fc both model")
        loc = self.model_directory + "/hcnn_fc_both.h5"
        self.fc.save(loc)
        return loc

    def save_cc_model(self):
        logger.info(f"Saving cc model")
        loc = self.model_directory + f"/hcnn_cc_epochs_tmp.h5"
        self.cc.save(loc)
        return loc

    def save_fc_model(self):
        logger.info(f"Saving fc model")
        loc = self.model_directory + f"/hcnn_fc_epochs_tmp.h5"
        self.fc.save(loc)
        return loc

    def load_best_cc_model(self):
        logger.info(f"Loading best cc model")
        self.load_cc_model(self.model_directory + "/hcnn_cc.h5")

    def load_best_fc_model(self):
        logger.info(f"Loading best fc model")
        self.load_fc_model(self.model_directory + "/hcnn_fc.h5")

    def load_best_cc_both_model(self):
        logger.info(f"Loading best cc both model")
        self.load_cc_model(self.model_directory + "/hcnn_cc_both.h5")

    def load_best_fc_both_model(self):
        logger.info(f"Loading best fc both model")
        self.load_fc_model(self.model_directory + "/hcnn_fc_both.h5")

    def load_cc_model(self, location):
        logger.info(f"Loading cc model")
        self.cc = tf.keras.models.load_model(location)

    def load_fc_model(self, location):
        logger.info(f"Loading fc model")
        self.fc = tf.keras.models.load_model(location, custom_objects={"SelfAttention": SelfAttention, "NormL": NormL})

    def train_coarse(self, training_data, validation_data, fine2coarse):
        x_train, y_train = training_data
        yc_train = tf.linalg.matmul(y_train, fine2coarse)

        x_val, y_val = validation_data
        yc_val = tf.linalg.matmul(y_val, fine2coarse)

        del y_train, y_val

        p = self.training_params
        val_thresh = p["validation_loss_threshold"]

        logger.debug(f"Creating coarse classifier with shared layers")
        self.cc, _ = self.build_cc_fc(verbose=False)
        self.fc = None
        optim = tf.keras.optimizers.SGD(lr=p['lr_coarse'], nesterov=True, momentum=0.5)

        loc = self.save_cc_model()

        logger.info('Start Coarse Classification Training')

        index = p['initial_epoch']

        prev_val_loss = float('inf')
        counts_patience = 0
        patience = p["patience"]
        while index < p['stop']:
            tf.keras.backend.clear_session()
            self.load_cc_model(loc)

            cc = tf.keras.Model(inputs=self.cc.inputs, outputs=self.cc.outputs[1])
            cc.compile(optimizer=optim,
                       loss='categorical_crossentropy',
                       metrics=['accuracy'])

            x_train, yc_train, _ = shuffle_data((x_train, yc_train))
            cc_fit = cc.fit(x_train, yc_train,
                            batch_size=p['batch_size'],
                            initial_epoch=index,
                            epochs=index + p["step"],
                            validation_data=(x_val, yc_val),
                            callbacks=[self.tbCallback_coarse])
            val_loss = cc_fit.history["val_loss"][-1]
            loc = self.save_cc_model()
            if prev_val_loss - val_loss < val_thresh:
                counts_patience += 1
                logger.info(f"Counts to early stopping: {counts_patience}/{p['patience']}")
                if counts_patience >= patience:
                    break
                elif counts_patience % p["reduce_lr_after_patience_counts"] == 0:
                    new_val = optim.learning_rate * p["lr_reduction_factor"]
                    logger.info(f"LR is now: {new_val.numpy()}")
                    optim.learning_rate.assign(new_val)
                    self.load_best_cc_model()
                    self.save_cc_model()
            else:
                counts_patience = 0
                prev_val_loss = val_loss
                self.save_best_cc_model()
            index += p["step"]

    def train_fine(self, training_data, validation_data, fine2coarse):
        x_train, y_train = training_data
        yc_train = tf.linalg.matmul(y_train, fine2coarse)
        x_val, y_val = validation_data
        yc_val = tf.linalg.matmul(y_val, fine2coarse)

        p = self.training_params
        val_thresh = p["validation_loss_threshold"]

        self.cc, self.fc = self.build_cc_fc(verbose=False)

        optim = tf.keras.optimizers.SGD(lr=p['lr_fine'], nesterov=True, momentum=0.5)

        loc_fc = self.save_fc_model()

        logger.info('Start Fine Classification Training')

        index = p['initial_epoch']

        prev_val_loss = float('inf')
        counts_patience = 0
        patience = p["patience"]

        while index < p['stop']:
            tf.keras.backend.clear_session()

            self.load_fc_model(loc_fc)
            self.load_best_cc_model()

            self.build_fine_model()

            x_train, y_train, inds = shuffle_data((x_train, y_train))
            yc_train = tf.gather(yc_train, inds)

            for l in self.cc.layers:
                l.trainable = False
            for l in self.fc.layers:
                l.trainable = True

            self.full_model.compile(optimizer=optim,
                                    loss='categorical_crossentropy',
                                    metrics=['accuracy'])

            fc_fit = self.full_model.fit([x_train, yc_train], y_train,
                                         batch_size=p['batch_size'],
                                         initial_epoch=index,
                                         epochs=index + p["step"],
                                         validation_data=([x_val, yc_val], y_val),
                                         callbacks=[self.tbCallback_fine])
            val_loss = fc_fit.history["val_loss"][-1]
            loc_fc = self.save_fc_model()
            if prev_val_loss - val_loss < val_thresh:
                counts_patience += 1
                logger.info(f"Counts to early stopping: {counts_patience}/{p['patience']}")
                if counts_patience >= patience:
                    break
                elif counts_patience % p["reduce_lr_after_patience_counts"] == 0:
                    new_val = optim.learning_rate * p["lr_reduction_factor"]
                    logger.info(f"LR is now: {new_val.numpy()}")
                    optim.learning_rate.assign(new_val)
                    self.load_best_fc_model()
                    self.save_fc_model()
            else:
                counts_patience = 0
                prev_val_loss = val_loss
                self.save_best_fc_model()
            index += p["step"]

    def train_both(self, training_data, validation_data, fine2coarse):
        x_train, y_train = training_data
        x_val, y_val = validation_data
        yc_train = tf.linalg.matmul(y_train, fine2coarse)
        yc_val = tf.linalg.matmul(y_val, fine2coarse)

        p = self.training_params
        val_thresh = p["validation_loss_threshold"]

        logger.info('Start Full Classification training')

        index = p['initial_epoch']

        tf.keras.backend.clear_session()
        self.load_best_cc_model()
        self.load_best_fc_model()
        loc_cc = self.save_cc_model()
        loc_fc = self.save_fc_model()

        tf.keras.backend.clear_session()

        optim = tf.keras.optimizers.SGD(lr=p['lr_full'], nesterov=True, momentum=0.5)

        prev_val_loss = float('inf')
        counts_patience = 0
        patience = p["patience"]
        while index < p['stop']:
            tf.keras.backend.clear_session()
            self.load_cc_model(loc_cc)
            self.load_fc_model(loc_fc)
            self.build_full_model()
            for l in self.cc.layers:
                l.trainable = True
            for l in self.fc.layers:
                l.trainable = True
            self.full_model.compile(optimizer=optim,
                                    loss='categorical_crossentropy',
                                    metrics=['accuracy'])
            x_train, y_train, inds = shuffle_data((x_train, y_train))
            yc_train = tf.gather(yc_train, inds)
            full_fit = self.full_model.fit(x_train, [y_train, yc_train],
                                           batch_size=p['batch_size'],
                                           initial_epoch=index,
                                           epochs=index + p["step_full"],
                                           validation_data=(x_val, [y_val, yc_val]),
                                           callbacks=[self.tbCallback_full])
            val_loss = full_fit.history["val_loss"][-1]
            loc_cc = self.save_cc_model()
            loc_fc = self.save_fc_model()
            if prev_val_loss - val_loss < val_thresh:
                counts_patience += 1
                logger.info(f"Counts to early stopping: {counts_patience}/{p['patience']}")
                if counts_patience >= patience:
                    break
                elif counts_patience % p["reduce_lr_after_patience_counts"] == 0:
                    new_val = optim.learning_rate * p["lr_reduction_factor"]
                    logger.info(f"LR is now: {new_val.numpy()}")
                    optim.learning_rate.assign(new_val)
                    self.load_best_fc_both_model()
                    self.save_fc_model()
                    self.load_best_cc_both_model()
                    self.save_cc_model()
            else:
                counts_patience = 0
                prev_val_loss = val_loss
                self.save_best_cc_both_model()
                self.save_best_fc_both_model()
            index += p["step_full"]

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

    def predict_full(self, testing_data, fine2coarse, results_file):
        x_test, y_test = testing_data
        yc_test = tf.linalg.matmul(y_test, fine2coarse)

        p = self.prediction_params

        self.load_best_cc_both_model()
        self.load_best_fc_both_model()
        self.build_full_model()

        [yh_s, ych_s] = self.full_model.predict(x_test, batch_size=p['batch_size'])

        fine_classification_error = utils.get_error(y_test, yh_s)
        logger.info('Fine Classifier Error: ' + str(fine_classification_error))

        coarse_classification_error = utils.get_error(yc_test, ych_s)
        logger.info('Coarse Classifier Error: ' + str(coarse_classification_error))

        mismatch = self.find_mismatch_error(yh_s, ych_s, fine2coarse)
        logger.info('Mismatch Error: ' + str(mismatch))

        results_dict = {'Fine Classifier Error': fine_classification_error,
                        'Coarse Classifier Error': coarse_classification_error,
                        'Mismatch Error': mismatch}

        self.write_results(results_file, results_dict=results_dict)

        np.save(self.model_directory + "/fine_predictions.npy", yh_s)
        np.save(self.model_directory + "/coarse_predictions.npy", ych_s)
        np.save(self.model_directory + "/fine_labels.npy", y_test)
        np.save(self.model_directory + "/coarse_labels.npy", yc_test)

        tf.keras.backend.clear_session()
        return yh_s, ych_s

    def predict_full_using_best_non_both(self, testing_data, fine2coarse, results_file):
        x_test, y_test = testing_data
        yc_test = tf.linalg.matmul(y_test, fine2coarse)

        p = self.prediction_params

        self.load_best_cc_model()
        self.load_best_fc_model()
        self.build_full_model()

        [yh_s, ych_s] = self.full_model.predict(x_test, batch_size=p['batch_size'])

        fine_classification_error = utils.get_error(y_test, yh_s)
        logger.info('Fine Classifier Error: ' + str(fine_classification_error))

        coarse_classification_error = utils.get_error(yc_test, ych_s)
        logger.info('Coarse Classifier Error: ' + str(coarse_classification_error))

        mismatch = self.find_mismatch_error(yh_s, ych_s, fine2coarse)
        logger.info('Mismatch Error: ' + str(mismatch))

        results_dict = {'Fine Classifier Error': fine_classification_error,
                        'Coarse Classifier Error': coarse_classification_error,
                        'Mismatch Error': mismatch}

        self.write_results(results_file, results_dict=results_dict)

        np.save(self.model_directory + "/fine_predictions.npy", yh_s)
        np.save(self.model_directory + "/coarse_predictions.npy", ych_s)
        np.save(self.model_directory + "/fine_labels.npy", y_test)
        np.save(self.model_directory + "/coarse_labels.npy", yc_test)

        tf.keras.backend.clear_session()
        return yh_s, ych_s

    def find_mismatch_error(self, fine_pred, coarse_pred, fine2coarse):
        # Convert fine pred to coarse pred
        coarse_pred_from_fine = tf.linalg.matmul(fine_pred, fine2coarse)
        n_pred = coarse_pred.shape[0]
        # Convert probabilities to labels
        c_l = np.argmax(coarse_pred, axis=1)
        cf_l = np.argmax(coarse_pred_from_fine, axis=1)
        # Find mismatches
        diff = np.where(c_l != cf_l)[0]
        mis = diff.shape[0] / n_pred
        return mis

    def write_results(self, results_file, results_dict):
        for a, b in results_dict.items():
            # Ensure that results_dict is made by numbers and lists only
            if type(b) is np.ndarray:
                results_dict[a] = b.tolist()
        json.dump(results_dict, open(results_file, 'w'))

    def build_cc_fc(self, verbose=True):
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

        # CC Output
        cc = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(cc_att)
        cc = tf.keras.layers.Flatten()(cc)
        cc = tf.keras.layers.Dense(512, activation='relu')(cc)
        cc = tf.keras.layers.Dropout(0.3)(cc)
        coarse = tf.keras.layers.Dense(self.n_coarse_categories, activation='softmax')(cc)

        # Build CC
        cc_model = tf.keras.Model(inputs=inp, outputs=[cc_att, coarse])
        if verbose:
            print(cc_model.summary())
        # FC Input
        fc_in_1 = tf.keras.Input(shape=cc_att.shape[1:])
        fc_in_2 = tf.keras.Input(shape=self.n_coarse_categories)

        # FC Model
        fc = tf.keras.layers.Conv2D(128, kernel_size, strides=(1, 1), padding='same')(fc_in_1)
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
        fc_out = tf.keras.layers.concatenate([fc_flat_out, fc_in_2])
        fc_out = tf.keras.layers.Dense(256, activation='relu')(fc_out)
        fc_out = tf.keras.layers.Dropout(0.3)(fc_out)
        fc_out = tf.keras.layers.Dense(self.n_fine_categories, activation='softmax')(fc_out)

        # Build FC
        fc_model = tf.keras.models.Model(inputs=[fc_in_1, fc_in_2], outputs=fc_out)
        if verbose:
            print(fc_model.summary())

        return cc_model, fc_model

    def build_full_model(self):
        inp = tf.keras.Input(shape=self.cc.input.shape[1:])
        cc_feat, cc_lab = self.cc(inp)

        fc_lab = self.fc([cc_feat, cc_lab])
        self.full_model = tf.keras.Model(inputs=inp, outputs=[fc_lab, cc_lab])

    def build_fine_model(self):
        inp2 = tf.keras.Input(shape=self.cc.outputs[1].shape[1:])
        inp = tf.keras.Input(shape=self.cc.input.shape[1:])
        s = self.cc.outputs[0].shape

        cc_feat, _ = self.cc(inp)
        fc_lab = self.fc([cc_feat, inp2])
        self.full_model = tf.keras.Model(inputs=[inp, inp2], outputs=fc_lab)
