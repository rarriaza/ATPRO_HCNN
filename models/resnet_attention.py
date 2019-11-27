import datetime
import json
import logging

import numpy as np
import tensorflow as tf

import utils
from models.plugins.model_saver import ModelSaver
from models.resnet_common import ResNet50

logger = logging.getLogger('ResNetBaseline')


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

        self.cc, self.fc = None, None
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

        self.tbCallback = tf.keras.callbacks.TensorBoard(
            log_dir=logs_directory + '/' + current_time,
            update_freq='epoch')  # How often to write logs (default: once per epoch)
        # self.early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=5)
        # self.reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_accuracy', factor=0.2,
        #                                                       patience=5, min_lr=0.0000001)

        att_input = tf.keras.Input(shape=[8, 8, 256])
        att_output = self.build_attention(att_input)
        self.attention = tf.keras.models.Model(inputs=att_input, outputs=att_output)

        # self.model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
        #     filepath=self.model_directory + "/resnet_attention_{epoch:02d}-epochs.h5",
        #     monitor='val_accuracy',
        #     save_best_only=True,
        #     mode='auto',
        #     save_freq=90000,
        #     verbose=1)

        self.training_params = {
            'batch_size': 64,
            'initial_epoch': 0,
            'lr_coarse': 3e-5,
            'lr_fine': 1e-5,
            'step': 1,  # Save weights every this amount of epochs
            'stop': 100,
            'patience': 3,
            'patience_decrement': 10,
            'decrement_lr': 0.2
        }

        self.prediction_params = {
            'batch_size': 64
        }

    def save_cc_model(self, epochs, val_accuracy, learning_rate):
        logger.info(f"Saving cc model")
        self.cc.save(
            self.model_directory + f"/resnet_attention_cc_epochs_{epochs:02d}_valacc_{val_accuracy:.4}_lr_{learning_rate:.4}.h5")

    def save_fc_model(self, epochs, val_accuracy, learning_rate):
        logger.info(f"Saving fc model")
        self.fc.save(
            self.model_directory + f"/resnet_attention_fc_epochs_{epochs:02d}_valacc_{val_accuracy:.4}_lr_{learning_rate:.4}.h5")

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

        logger.info('Start Coarse Classification Training')

        adam_coarse = tf.keras.optimizers.Adam(lr=p['lr_coarse'])
        self.cc.compile(optimizer=adam_coarse,
                        loss='categorical_crossentropy',
                        metrics=['accuracy'])
        index = p['initial_epoch']

        prev_val_acc = 0.0
        counts_patience = 0
        patience = p["patience"]
        decremented = 0
        while index < p['stop']:
            cc_fit = self.cc.fit(x_train, yc_train,
                                 batch_size=p['batch_size'],
                                 initial_epoch=index,
                                 epochs=index + p["step"],
                                 validation_data=(x_val, yc_val),
                                 callbacks=[self.tbCallback])
            val_acc = cc_fit.history["val_accuracy"][-1]
            if val_acc - prev_val_acc < 0:
                counts_patience += 1
                if counts_patience >= patience:
                    self.save_cc_model(index, val_acc, self.cc.optimizer.learning_rate.numpy())
                    break
                else:
                    # Decrement LR
                    decremented += 1
                    logger.info(
                        f"Decreasing learning rate from {self.cc.optimizer.learning_rate.numpy()} to {self.cc.optimizer.learning_rate.numpy() * p['decrement_lr']}")
                    self.cc.optimizer.learning_rate.assign(self.cc.optimizer.learning_rate * p['decrement_lr'])
            else:
                counts_patience = 0
            if decremented >= p['patience_decrement']:
                self.save_cc_model(index, val_acc, self.cc.optimizer.learning_rate.numpy())
                break
            index += p["step"]
            self.save_cc_model(index, val_acc, self.cc.optimizer.learning_rate.numpy())
            prev_val_acc = val_acc

    def train_fine(self, training_data, validation_data, fine2coarse):
        x_train, y_train = training_data
        yc_train = tf.linalg.matmul(y_train, fine2coarse)
        x_val, y_val = validation_data
        yc_val = tf.linalg.matmul(y_val, fine2coarse)

        p = self.training_params

        logger.debug(f"Creating fine classifier with shared layers")
        self.cc, self.fc = self.build_cc_fc()

        logger.info("Attention reweighting of training features")
        feature_model = tf.keras.models.Model(inputs=self.cc.input,
                                              outputs=self.cc.get_layer('conv2_block3_out').output)
        feature_map = feature_model.predict(x_train, batch_size=p['batch_size'])
        feature_map_att = self.attention.predict(feature_map, batch_size=p['batch_size'])
        logger.info("Attention reweighting of validation features")
        feature_map = feature_model.predict(x_val, batch_size=p['batch_size'])
        feature_map_att_val = self.attention.predict(feature_map, batch_size=p['batch_size'])

        del feature_map

        logger.info('Start Fine Classification Training')

        adam_fine = tf.keras.optimizers.Adam(lr=p['lr_fine'])
        self.fc.compile(optimizer=adam_fine,
                        loss='categorical_crossentropy',
                        metrics=['accuracy'])
        index = p['initial_epoch']

        prev_val_acc = 0.0
        counts_patience = 0
        patience = p["patience"]
        decremented = 0
        while index < p['stop']:
            fc_fit = self.fc.fit([feature_map_att, yc_train], y_train,
                                 batch_size=p['batch_size'],
                                 initial_epoch=index,
                                 epochs=index + p["step"],
                                 validation_data=([feature_map_att_val, yc_val], y_val),
                                 callbacks=[self.tbCallback])
            val_acc = fc_fit.history["val_accuracy"][-1]
            if val_acc - prev_val_acc < 0:
                if counts_patience == 0:
                    best_epoch = index - p['step']
                counts_patience += 1
                if counts_patience >= patience:
                    self.save_fc_model(index, val_acc, self.fc.optimizer.learning_rate.numpy())
                    break
                else:
                    # Decrement LR
                    decremented += 1
                    logger.info(
                        f"Decreasing learning rate from {self.fc.optimizer.learning_rate.numpy()} to {self.fc.optimizer.learning_rate.numpy() * p['decrement_lr']}")
                    self.fc.optimizer.learning_rate.assign(self.fc.optimizer.learning_rate * p['decrement_lr'])
            else:
                counts_patience = 0
            if decremented >= p['patience_decrement']:
                self.save_fc_model(index, val_acc, self.fc.optimizer.learning_rate.numpy())
                break
            index += p["step"]
            self.save_fc_model(index, val_acc, self.fc.optimizer.learning_rate.numpy())
            prev_val_acc = val_acc

        logger.info(f"Best model generated on epoch: {best_epoch}")

    def predict_coarse(self, testing_data, fine2coarse, results_file):
        x_test, y_test = testing_data
        yc_test = tf.linalg.matmul(y_test, fine2coarse)

        p = self.prediction_params

        yc_pred = self.cc.predict(x_test, batch_size=p['batch_size'])

        single_classifier_error = utils.get_error(yc_test, yc_pred)
        logger.info('Single Classifier Error: ' + str(single_classifier_error))

        coarse_classifier_error = utils.get_error(yc_test, yc_pred)

        logger.info('Single Classifier Error: ' + str(coarse_classifier_error))
        results_dict = {'Single Classifier Error': single_classifier_error,
                        'Coarse Classifier Error': coarse_classifier_error}
        self.write_results(results_file, results_dict=results_dict)

        return yc_pred

    def predict_fine(self, testing_data, yc_pred, results_file):
        x_test, y_test = testing_data

        features_test = self.build_attention(x_test)

        p = self.prediction_params

        yh_s = self.fc.predict([features_test, yc_pred], batch_size=p['batch_size'])

        single_classifier_error = utils.get_error(y_test, yh_s)
        logger.info('Single Classifier Error: ' + str(single_classifier_error))

        results_dict = {'Single Classifier Error': single_classifier_error}
        self.write_results(results_file, results_dict=results_dict)

        return yh_s

    def write_results(self, results_file, results_dict):
        for a, b in results_dict.items():
            # Ensure that results_dict is made by numbers and lists only
            if type(b) is np.ndarray:
                results_dict[a] = b.tolist()
        json.dump(results_dict, open(results_file, 'w'))

    def build_cc_fc(self):
        model_1, model_2 = ResNet50(include_top=False, weights='imagenet',
                                    input_tensor=None, input_shape=self.input_shape,
                                    pooling=None, classes=1000)

        # Define CC Prediction Block
        cc_flat = tf.keras.layers.Flatten()(model_1.output)
        cc_out = tf.keras.layers.Dense(
            self.n_coarse_categories, activation='softmax')(cc_flat)

        cc_model = tf.keras.models.Model(inputs=model_1.input, outputs=cc_out)
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
        print(fc_model.summary())

        return cc_model, fc_model

    def build_attention(self, inp):
        weights = tf.reduce_sum(inp, axis=(1, 2))
        weights = tf.math.l2_normalize(weights, axis=1)
        weights = tf.expand_dims(weights, axis=1)
        weights = tf.expand_dims(weights, axis=1)
        weigthed_channels = tf.multiply(inp, weights)
        attention_map = tf.expand_dims(tf.reduce_sum(weigthed_channels, 3), 3)
        cropped_features = tf.multiply(inp, attention_map)
        return cropped_features
