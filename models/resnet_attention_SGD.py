import datetime
import json
import logging

import numpy as np
import tensorflow as tf
from random import randint
from tensorflow.keras.layers import Lambda, Reshape, Add
from tensorflow.keras import Layer

import utils
from datasets.preprocess import shuffle_data
from models.resnet_common import ResNet50

logger = logging.getLogger('ResNetAttention')


class ResNetAttention:
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

        self.attention_input_shape = [8, 8, 256]  # NICE-TO-HAVE: this shouldn't be hardcoded

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
            'lr_coarse': 3e-6,
            'lr_fine': 3e-6,
            'lr_full': 1e-7,
            'step': 5,  # Save weights every this amount of epochs
            'step_full': 1,
            'stop': 10000,
            'patience': 5
        }

        if self.args.debug_mode:
            self.training_params['step'] = 1
            self.training_params['stop'] = 1

        self.prediction_params = {
            'batch_size': 64
        }

    def save_best_cc_model(self):
        logger.info(f"Saving best cc model")
        loc = self.model_directory + "/resnet_attention_cc_SGD.h5"
        self.cc.save(loc)
        return loc

    def save_best_fc_model(self):
        logger.info(f"Saving best fc model")
        loc = self.model_directory + "/resnet_attention_fc_SGD.h5"
        self.fc.save(loc)
        return loc

    def save_best_cc_both_model(self):
        logger.info(f"Saving best cc both model")
        loc = self.model_directory + "/resnet_attention_cc_both_SGD.h5"
        self.cc.save(loc)
        return loc

    def save_best_fc_both_model(self):
        logger.info(f"Saving best fc both model")
        loc = self.model_directory + "/resnet_attention_fc_both_SGD.h5"
        self.fc.save(loc)
        return loc

    def save_cc_model(self, epochs, val_accuracy):
        logger.info(f"Saving cc model")
        loc = self.model_directory + f"/resnet_attention_cc_epochs_{epochs:02d}_valacc_{val_accuracy:.4}_SGD.h5"
        self.cc.save(loc)
        return loc

    def save_fc_model(self, epochs, val_accuracy):
        logger.info(f"Saving fc model")
        loc = self.model_directory + f"/resnet_attention_fc_epochs_{epochs:02d}_valacc_{val_accuracy:.4}_SGD.h5"
        self.fc.save(loc)
        return loc

    def save_cc_both_model(self, epochs, val_accuracy):
        logger.info(f"Saving cc both model")
        loc = self.model_directory + f"/resnet_attention_cc_both_epochs_{epochs:02d}_valacc_{val_accuracy:.4}_SGD.h5"
        self.cc.save(loc)
        return loc

    def save_fc_both_model(self, epochs, val_accuracy):
        logger.info(f"Saving fc both model")
        loc = self.model_directory + f"/resnet_attention_fc_both_epochs_{epochs:02d}_valacc_{val_accuracy:.4}_SGD.h5"
        self.fc.save(loc)
        return loc

    def load_best_cc_model(self):
        logger.info(f"Loading best cc model")
        self.load_cc_model(self.model_directory + "/resnet_attention_cc_SGD.h5")

    def load_best_fc_model(self):
        logger.info(f"Loading best fc model")
        self.load_fc_model(self.model_directory + "/resnet_attention_fc_SGD.h5")

    def load_best_cc_both_model(self):
        logger.info(f"Loading best cc both model")
        self.load_cc_model(self.model_directory + "/resnet_attention_cc_both_SGD.h5")

    def load_best_fc_both_model(self):
        logger.info(f"Loading best fc both model")
        self.load_fc_model(self.model_directory + "/resnet_attention_fc_both_SGD.h5")

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
        SGD_coarse = tf.keras.optimizers.SGD(learning_rate=p['lr_coarse'], momentum=0.6, nesterov=False)
        self.cc.compile(optimizer=SGD_coarse,
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
        SGD_fine = tf.keras.optimizers.SGD(learning_rate=p['lr_fine'], momentum=0.6, nesterov=False)
        self.fc.compile(optimizer=SGD_fine,
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
        SGD_fine = tf.keras.optimizers.SGD(learning_rate=p['lr_full'], momentum=0.6, nesterov=False)
        self.full_model.compile(optimizer=SGD_fine,
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
            adam_fine = tf.keras.optimizers.Adam(lr=p['lr_fine'])
            SGD_fine = tf.keras.optimizers.SGD(learning_rate=p['lr_fine'], momentum=0.6, nesterov=False)
            self.full_model.compile(optimizer=SGD_fine,
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
            val_acc_coarse = full_fit.history["val_dense_accuracy"][-1]
            val_loss = full_fit.history["val_loss"][-1]
            val_loss_coarse = full_fit.history["val_dense_loss"][-1]
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

    def build_attention(self, l=8*8, d=256, dv=64, dout=256, nv=8):

        att_input1 = tf.keras.Input(shape=self.attention_input_shape)
        att_input2 = tf.keras.Input(shape=self.attention_input_shape)
        att_input3 = tf.keras.Input(shape=self.attention_input_shape)

        att1 = Reshape([l, d])(att_input1)
        att2 = Reshape([l, d])(att_input2)
        att3 = Reshape([l, d])(att_input3)

        dense1 = tf.keras.layers.Dense(dv * nv, activation='relu')(att1)
        dense2 = tf.keras.layers.Dense(dv * nv, activation='relu')(att2)
        dense3 = tf.keras.layers.Dense(dv * nv, activation='relu')(att3)

        resh1 = Reshape([l, nv, dv])(dense1)
        resh2 = Reshape([l, nv, dv])(dense2)
        resh3 = Reshape([l, nv, dv])(dense3)

        att = Lambda(lambda x: tf.keras.backend.batch_dot(x[0], x[1], axes=[-1, -1])/np.sqrt(dv),
                     output_shape=(l, nv, nv))([resh2, resh3])
        att = Lambda(lambda x: tf.keras.backend.softmax(x) / np.sqrt(dv),
                     output_shape=(l, nv, nv))(att)

        out = Lambda(lambda x: tf.keras.backend.batch_dot(x[0], x[1], axes=[4, 3]),
                     output_shape=(l, nv, nv))([att, resh1])
        out = Reshape([l, d])(out)

        out = Add()([out, att2])

        out = tf.keras.layers.Dense(dout * l, activation= 'relu')(out)

        feature_map_att = Reshape([8, 8, 256])(out)

        return tf.keras.Model(inputs=[att_input2, att_input3, att_input1], outputs=feature_map_att)

    def get_feature_input_for_fc(self, data):
        batch_size = self.prediction_params['batch_size']
        self.cc, _ = self.build_cc_fc(verbose=False)

        self.load_best_cc_model()

        self.attention = self.build_attention()

        feature_model = tf.keras.models.Model(inputs=self.cc.input,
                                              outputs=self.cc.get_layer('conv2_block3_out').output)
        feature_map = feature_model.predict(data, batch_size=batch_size)
        feature_map_att = self.attention.predict(feature_map, batch_size=batch_size)

        tf.keras.backend.clear_session()
        return feature_map_att

    def build_full_model(self):
        att_mod = self.build_attention()
        cc_mod_feat = tf.keras.Model(self.cc.input, [self.cc.layers[-3].output, self.cc.output])
        cc_mod_feat._name = "dont_care"
        cc_feat = cc_mod_feat(self.cc.input)
        att_out = att_mod(cc_feat[0])
        fc_out = self.fc([att_out, self.cc.output])
        self.full_model = tf.keras.Model(inputs=self.cc.inputs, outputs=[fc_out, self.cc.output])

class NormL(Layer):
    def __init__(self, **kwargs):
        super(NormL, self).__init__(**kwargs)

    def build(self, input_shape):
        self.a = self.add_weight(name='kernel',
                                 shape=(1, input_shape[-1]),
                                 initializer='ones',
                                 trainable=True)
        self.b = self.add_weight(name='kernel',
                                 shape=(1, input_shape[-1]),
                                 initializer='zeros',
                                 trainable=True)
        super(NormL, self).build(input_shape)

    def call(self, x):
        eps = 0.000001
        mu = tf.keras.backend.mean(x, keepdims=True, axis=-1)
        sigma = tf.keras.backend.std(x, keepdims=True, axis=-1)
        ln_out = (x - mu)/(sigma + eps)
        return ln_out * self.a + self.b
