import datetime
import json
import logging

import numpy as np
import tensorflow as tf
from random import randint

import utils
from datasets.preprocess import shuffle_data

logger = logging.getLogger('HAT-CNN')


class HatCNN:
    def __init__(self, n_fine_categories, n_coarse_categories, input_shape,
                 logs_directory=None, model_directory=None, args=None):
        """
        HAT CNN
        """
        self.model_directory = model_directory
        self.args = args
        self.n_fine_categories = n_fine_categories
        self.n_coarse_categories = n_coarse_categories
        self.input_shape = input_shape

        self.cc, self.fc, self.full_model = None, None, None
        self.attention = None

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
            'lr_coarse': 0.001,
            'lr_fine': 0.0001,
            'lr_full': 1e-6,
            'step': 5,  # Save weights every this amount of epochs
            'step_full': 1,
            'stop': 10000,
            'patience': 5
        }

        self.attention_units = 128
        self.attention_input_shape = [8, 8, self.attention_units]

        if self.args.debug_mode:
            self.training_params['step'] = 1
            self.training_params['stop'] = 1

        self.prediction_params = {
            'batch_size': 64
        }

    def save_best_cc_model(self):
        logger.info(f"Saving best cc model")
        loc = self.model_directory + "/resnet_attention_cc.h5"
        self.cc.save(loc)
        return loc

    def save_best_fc_model(self):
        logger.info(f"Saving best fc model")
        loc = self.model_directory + "/resnet_attention_fc.h5"
        self.fc.save(loc)
        return loc

    def save_best_cc_both_model(self):
        logger.info(f"Saving best cc both model")
        loc = self.model_directory + "/resnet_attention_cc_both.h5"
        self.cc.save(loc)
        return loc

    def save_best_fc_both_model(self):
        logger.info(f"Saving best fc both model")
        loc = self.model_directory + "/resnet_attention_fc_both.h5"
        self.fc.save(loc)
        return loc

    def save_cc_model(self, epochs, val_accuracy):
        logger.info(f"Saving cc model")
        loc = self.model_directory + f"/resnet_attention_cc_epochs_{epochs:02d}_valacc_{val_accuracy:.4}.h5"
        self.cc.save(loc)
        return loc

    def save_fc_model(self, epochs, val_accuracy):
        logger.info(f"Saving fc model")
        loc = self.model_directory + f"/resnet_attention_fc_epochs_{epochs:02d}_valacc_{val_accuracy:.4}.h5"
        self.fc.save(loc)
        return loc

    def save_cc_both_model(self, epochs, val_accuracy):
        logger.info(f"Saving cc both model")
        loc = self.model_directory + f"/resnet_attention_cc_both_epochs_{epochs:02d}_valacc_{val_accuracy:.4}.h5"
        self.cc.save(loc)
        return loc

    def save_fc_both_model(self, epochs, val_accuracy):
        logger.info(f"Saving fc both model")
        loc = self.model_directory + f"/resnet_attention_fc_both_epochs_{epochs:02d}_valacc_{val_accuracy:.4}.h5"
        self.fc.save(loc)
        return loc

    def load_best_cc_model(self):
        logger.info(f"Loading best cc model")
        self.load_cc_model(self.model_directory + "/resnet_attention_cc.h5")

    def load_best_fc_model(self):
        logger.info(f"Loading best fc model")
        self.load_fc_model(self.model_directory + "/resnet_attention_fc.h5")

    def load_best_cc_both_model(self):
        logger.info(f"Loading best cc both model")
        self.load_cc_model(self.model_directory + "/resnet_attention_cc_both.h5")

    def load_best_fc_both_model(self):
        logger.info(f"Loading best fc both model")
        self.load_fc_model(self.model_directory + "/resnet_attention_fc_both.h5")

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
        prev_val_loss = float('inf')
        val_loss = 0
        counts_patience = 0
        patience = p["patience"]
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
            if prev_val_loss - val_loss < 5e-3:
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

        feature_map_att = self.get_feature_input_for_fc(x_train)
        feature_map_att_val = self.get_feature_input_for_fc(x_val)

        _, self.fc = self.build_cc_fc(verbose=False)
        adam_fine = tf.keras.optimizers.Adam(lr=p['lr_fine'])
        self.fc.compile(optimizer=adam_fine,
                        loss='categorical_crossentropy',
                        metrics=['accuracy'])
        loc = self.save_fc_model(0, 0.0)
        tf.keras.backend.clear_session()
        # self.load_fc_model(loc)

        logger.info('Start Fine Classification Training')

        index = p['initial_epoch']

        prev_val_loss = float('inf')
        val_loss = 0
        counts_patience = 0
        patience = p["patience"]
        best_model = loc
        while index < p['stop']:
            tf.keras.backend.clear_session()
            self.load_fc_model(loc)
            s = randint(0, 10000)
            feature_map_att, y_train, inds = shuffle_data((feature_map_att, y_train), random_state=s)
            yc_train = tf.gather(yc_train, inds)

            fc_fit = self.fc.fit([feature_map_att, yc_train], y_train,
                                 batch_size=p['batch_size'],
                                 initial_epoch=index,
                                 epochs=index + p["step"],
                                 validation_data=([feature_map_att_val, yc_val], y_val),
                                 callbacks=[self.tbCallback_fine])
            val_loss = fc_fit.history["val_loss"][-1]
            val_acc = fc_fit.history["val_accuracy"][-1]
            loc = self.save_fc_model(index, val_acc)
            if prev_val_loss - val_loss < 5e-3:
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
        loc_cc = "./saved_models/attention/resnet_attention_cc.h5"
        loc_fc = "./saved_models/attention/resnet_attention_fc.h5"
        self.load_cc_model(loc_cc)
        self.load_fc_model(loc_fc)

        self.build_full_model()
        adam_fine = tf.keras.optimizers.Adam(lr=p['lr_full'])
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

        prev_val_loss = float('inf')
        counts_patience = 0
        patience = p["patience"]
        while index < p['stop']:
            tf.keras.backend.clear_session()
            # self.load_full_model(loc)
            self.load_cc_model(loc_cc)
            self.load_fc_model(loc_fc)
            self.build_full_model()
            adam_fine = tf.keras.optimizers.Adam(lr=p['lr_full'])
            self.full_model.compile(optimizer=adam_fine,
                                    loss='categorical_crossentropy',
                                    metrics=['accuracy'])
            x_train, y_train, inds = shuffle_data((x_train, y_train))
            yc_train = tf.gather(yc_train, inds)
            full_fit = self.full_model.fit(x_train, [y_train, yc_train],
                                           batch_size=p['batch_size'],
                                           initial_epoch=index,
                                           epochs=index + p["step_full"],
                                           validation_data=(x_val, [y_val, yc_val]),
                                           callbacks=[self.tbCallback_coarse])

            val_acc_fine = full_fit.history["val_model_1_accuracy"][-1]
            val_acc_coarse = full_fit.history["val_dense_1_accuracy"][-1]
            val_loss = full_fit.history["val_loss"][-1]
            val_loss_coarse = full_fit.history["val_dense_1_loss"][-1]
            loc_cc = self.save_cc_both_model(index, val_acc_coarse)
            loc_fc = self.save_fc_both_model(index, val_acc_fine)
            if prev_val_loss - val_loss < 5e-3:
                if counts_patience == 0:
                    best_model_cc = loc_cc
                    best_model_fc = loc_fc
                counts_patience += 1
                logger.info(f"Counts to early stopping: {counts_patience}/{p['patience']}")
                if counts_patience >= patience:
                    break
            else:
                counts_patience = 0
                prev_val_loss = val_loss
            index += p["step_full"]
        if best_model_cc is not None and best_model_fc is not None:
            tf.keras.backend.clear_session()
            self.load_cc_model(best_model_cc)
            self.load_fc_model(best_model_fc)

        best_model_cc = self.save_best_cc_both_model()
        best_model_fc = self.save_best_fc_both_model()
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
        cc = tf.keras.layers.Dense(self.n_coarse_categories, activation='softmax')(cc)

        # Build CC
        cc_model = tf.keras.Model(inputs=inp, outputs=cc)
        if verbose:
            print(cc_model.summary())
        # FC Input
        fc_in_1 = tf.keras.Input(shape=self.attention_input_shape)
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

    def build_attention(self):
        att_input = tf.keras.Input(shape=self.attention_input_shape)
        att_output = self.compute_attention(att_input)
        attention_model = tf.keras.models.Model(inputs=att_input, outputs=att_output)
        return attention_model

    def get_feature_input_for_fc(self, data):
        batch_size = self.prediction_params['batch_size']
        self.cc, _ = self.build_cc_fc(verbose=False)

        self.load_best_cc_model()

        self.attention = self.build_attention()

        feature_model = tf.keras.models.Model(inputs=self.cc.input,
                                              outputs=self.cc.get_layer('attention_layer').output)
        feature_map = feature_model.predict(data, batch_size=batch_size)
        feature_map_att = self.attention.predict(feature_map, batch_size=batch_size)

        tf.keras.backend.clear_session()
        return feature_map_att

    def compute_attention(self, inp):
        logger.info('Building attention features')
        weights = tf.reduce_sum(inp, axis=(1, 2))
        weights = tf.math.l2_normalize(weights, axis=1)
        weights = tf.expand_dims(weights, axis=1)
        weights = tf.expand_dims(weights, axis=1)
        weigthed_channels = tf.multiply(inp, weights)
        attention_map = tf.expand_dims(tf.reduce_sum(weigthed_channels, 3), 3)
        cropped_features = tf.multiply(inp, attention_map)
        return cropped_features

    def build_full_model(self):
        att_mod = self.build_attention()
        cc_mod_feat = tf.keras.Model(self.cc.input, [self.cc.get_layer('attention_layer').output, self.cc.output])
        cc_mod_feat._name = "dont_care"
        cc_feat = cc_mod_feat(self.cc.input)
        att_out = att_mod(cc_feat[0])
        fc_out = self.fc([att_out, self.cc.output])
        self.full_model = tf.keras.Model(inputs=self.cc.inputs, outputs=[fc_out, self.cc.output])
