import datetime
import json
import logging

import numpy as np
import tensorflow as tf
from random import randint

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
        loc = self.model_directory + "/resnet_attention_cc.h5"
        self.cc.save(loc)
        return loc

    def save_best_fc_model(self):
        logger.info(f"Saving best fc model")
        loc = self.model_directory + "/resnet_attention_fc.h5"
        self.fc.save(loc)
        return loc

    def save_best_full_model(self):
        logger.info(f"Saving best full model")
        loc = self.model_directory + "/resnet_attention_full.h5"
        self.fc.save(loc)
        return loc

    def save_cc_model(self, epochs, val_accuracy, learning_rate):
        logger.info(f"Saving cc model")
        loc = self.model_directory + f"/resnet_attention_cc_epochs_{epochs:02d}_valacc_{val_accuracy:.4}_lr_{learning_rate:.4}.h5"
        self.cc.save(loc)
        return loc

    def save_fc_model(self, epochs, val_accuracy, learning_rate):
        logger.info(f"Saving fc model")
        loc = self.model_directory + f"/resnet_attention_fc_epochs_{epochs:02d}_valacc_{val_accuracy:.4}_lr_{learning_rate:.4}.h5"
        self.fc.save(loc)
        return loc

    def save_full_model(self, epochs, val_accuracy, learning_rate):
        logger.info(f"Saving full model")
        loc = self.model_directory + f"/resnet_attention_full_epochs_{epochs:02d}_valacc_{val_accuracy:.4}_lr_{learning_rate:.4}.h5"
        self.full_model.save(loc)
        return loc

    def load_best_cc_model(self):
        logger.info(f"Loading best cc model")
        self.load_cc_model(self.model_directory + "/resnet_attention_cc.h5")

    def load_best_fc_model(self):
        logger.info(f"Loading best fc model")
        self.load_fc_model(self.model_directory + "/resnet_attention_fc.h5")

    def load_cc_model(self, location):
        logger.info(f"Loading cc model")
        self.cc = tf.keras.models.load_model(location)

    def load_fc_model(self, location):
        logger.info(f"Loading fc model")
        self.fc = tf.keras.models.load_model(location)

    def load_full_model(self, location):
        logger.info(f"Loading full model")
        self.full_model = tf.keras.models.load_model(location)

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

        loc = self.save_cc_model(0, 0.0, p['lr_coarse'])

        logger.info('Start Coarse Classification Training')

        index = p['initial_epoch']

        best_model = loc
        prev_val_acc = 0.0
        val_acc = 0
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
            val_acc = cc_fit.history["val_accuracy"][-1]
            loc = self.save_cc_model(index, val_acc, self.cc.optimizer.learning_rate.numpy())
            if val_acc - prev_val_acc < 0:
                if counts_patience == 0:
                    best_model = loc
                counts_patience += 1
                logger.info(f"Counts to early stopping: {counts_patience}/{p['patience']}")
                if counts_patience >= patience:
                    break
                else:
                    pass
                    # Decrement LR
                    # logger.info(
                    #     f"Decreasing learning rate from {self.cc.optimizer.learning_rate.numpy()} to {self.cc.optimizer.learning_rate.numpy() * p['decrement_lr']}")
                    # self.cc.optimizer.learning_rate.assign(self.cc.optimizer.learning_rate * p['decrement_lr'])
            else:
                counts_patience = 0
                prev_val_acc = val_acc
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
        loc = self.save_fc_model(0, 0.0, 1e-5)
        tf.keras.backend.clear_session()
        # self.load_fc_model(loc)

        logger.info('Start Fine Classification Training')

        index = p['initial_epoch']

        prev_val_acc = 0.0
        val_acc = 0
        counts_patience = 0
        patience = p["patience"]
        decremented = 0
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
            val_acc = fc_fit.history["val_accuracy"][-1]
            loc = self.save_fc_model(index, val_acc, self.fc.optimizer.learning_rate.numpy())
            if val_acc - prev_val_acc < 0:
                if counts_patience == 0:
                    best_model = loc
                counts_patience += 1
                logger.info(f"Counts to early stopping: {counts_patience}/{p['patience']}")
                if counts_patience >= patience:
                    break
                else:
                    pass
                    # Decrement LR
                    # decremented += 1
                    # logger.info(
                    #     f"Decreasing learning rate from {self.fc.optimizer.learning_rate.numpy()} to {self.fc.optimizer.learning_rate.numpy() * p['decrement_lr']}")
                    # self.fc.optimizer.learning_rate.assign(self.fc.optimizer.learning_rate * p['decrement_lr'])
            else:
                counts_patience = 0
                prev_val_acc = val_acc
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
        loc_cc = "./saved_models/resnet_attention_cc.h5"
        loc_fc = "./saved_models/resnet_attention_fc.h5"
        self.load_best_cc_model()
        self.load_best_fc_model()


        att_mod = self.build_attention()
        cc_mod_feat = tf.keras.Model(self.cc.input, [self.cc.layers[-3].output, self.cc.output])
        cc_mod_feat._name = "dont_care"
        cc_feat = cc_mod_feat(self.cc.input)
        att_out = att_mod(cc_feat[0])
        fc_out = self.fc([att_out, self.cc.output])
        self.full_model = tf.keras.Model(inputs=self.cc.inputs, outputs=[fc_out, self.cc.output])
        adam_fine = tf.keras.optimizers.Adam(lr=p['lr_fine'])
        self.full_model.compile(optimizer=adam_fine,
                           loss='categorical_crossentropy',
                           metrics=['accuracy'])

        loc = self.save_full_model(0, 0.0, self.full_model.optimizer.learning_rate.numpy())
        best_model = loc
        self.load_full_model(loc)

        prev_val_acc = 0.0
        val_acc = 0
        counts_patience = 0
        patience = p["patience"]
        decremented = 0
        while index < p['stop']:
            tf.keras.backend.clear_session()
            self.load_full_model(loc)
            x_train, y_train, inds = shuffle_data((x_train, y_train))
            yc_train = tf.gather(yc_train, inds)
            cc_fit = self.full_model.fit(x_train, [y_train, yc_train],
                                    batch_size=p['batch_size'],
                                    initial_epoch=index,
                                    epochs=index + p["step"],
                                    validation_data=(x_val, [y_val, yc_val]),
                                    callbacks=[self.tbCallback_coarse])
            val_acc = cc_fit.history["val_accuracy"][-1]
            loc = self.save_full_model(index, val_acc, self.full_model.optimizer.learning_rate.numpy())
            if val_acc - prev_val_acc < 0:
                if counts_patience == 0:
                    best_model = loc
                counts_patience += 1
                logger.info(f"Counts to early stopping: {counts_patience}/{p['patience']}")
                if counts_patience >= patience:
                    break
                else:
                    pass
                    # Decrement LR
                    # logger.info(
                    #     f"Decreasing learning rate from {self.cc.optimizer.learning_rate.numpy()} to {self.cc.optimizer.learning_rate.numpy() * p['decrement_lr']}")
                    # self.cc.optimizer.learning_rate.assign(self.cc.optimizer.learning_rate * p['decrement_lr'])
            else:
                counts_patience = 0
                prev_val_acc = val_acc
            index += p["step"]
        if best_model is not None:
            tf.keras.backend.clear_session()
            self.load_full_model(best_model)

        best_model = self.save_best_full_model()
        # best_model = loc  This is just for debugging purposes
        return best_model

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

    def build_attention(self):
        att_input = tf.keras.Input(shape=self.attention_input_shape)
        att_output = self.compute_attention(att_input)
        attention_model = tf.keras.models.Model(inputs=att_input, outputs=att_output)
        return attention_model

    def get_feature_input_for_fc(self, data):
        batch_size = self.prediction_params['batch_size']
        self.cc, _ = self.build_cc_fc(verbose=False)
        self.attention = self.build_attention()

        feature_model = tf.keras.models.Model(inputs=self.cc.input,
                                              outputs=self.cc.get_layer('conv2_block3_out').output)
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
