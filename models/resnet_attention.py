import tensorflow as tf
import logging
import numpy as np

import utils
import json

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

        logger.debug(f"Creating full classifier with shared layers")
        self.cc, self.fc = self.build_cc_fc()

        self.tbCallBack = tf.keras.callbacks.TensorBoard(
            log_dir=logs_directory, histogram_freq=0,
            write_graph=True, write_images=True)

        self.training_params = {
            'batch_size': 64,
            'initial_epoch': 0,
            'lr_coarse': 0.001,
            'lr_decay_coarse': 1e-5,
            'lr_fine': 0.0001,
            'lr_decay_fine': 1e-6,
            'step': 5,  # Save weights every this amount of epochs
            'stop': 500
        }

        self.prediction_params = {
            'batch_size': 64
        }

    def train(self, training_data, validation_data, fine2coarse):
        x_train, y_train = training_data
        x_train, y_train = x_train, y_train
        yc_train = tf.tensordot(y_train, fine2coarse, 1)

        x_val, y_val = validation_data
        x_val, y_val = x_val, y_val
        yc_val = tf.tensordot(y_val, fine2coarse, 1)

        p = self.training_params

        logger.info('Start Coarse Classification Training')

        adam_coarse = tf.keras.optimizers.Adam(lr=p['lr_coarse'], decay=p['lr_decay_coarse'])
        self.cc.compile(optimizer=adam_coarse,
                        loss='categorical_crossentropy',
                        metrics=['accuracy'])
        index = p['initial_epoch']

        while index < p['stop']:
            self.cc.fit(x_train, yc_train,
                        batch_size=p['batch_size'],
                        initial_epoch=index,
                        epochs=index + p['step'],
                        validation_data=(x_val, yc_val),
                        callbacks=[self.tbCallBack])
            index += p['step']

        logger.info('Start Fine Classification Training')

        feature_map_att = self.attention(tf.cast(x_train, tf.dtypes.float32))
        feature_map_att_val = self.attention(tf.cast(x_val, tf.dtypes.float32))

        yc_pred = self.cc(tf.cast(x_train, tf.dtypes.float32))
        yc_val_pred = self.cc(tf.cast(x_val, tf.dtypes.float32))

        adam_fine = tf.keras.optimizers.Adam(lr=p['lr_fine'], decay=p['lr_decay_fine'])
        self.fc.compile(optimizer=adam_fine,
                        loss='categorical_crossentropy',
                        metrics=['accuracy'])
        index = p['initial_epoch']

        while index < p['stop']:
            self.fc.fit([feature_map_att, yc_pred], y_train,
                        batch_size=p['batch_size'],
                        initial_epoch=index,
                        epochs=index + p['step'],
                        validation_data=([feature_map_att_val, yc_val_pred], y_val),
                        callbacks=[self.tbCallBack])
            index += p['step']


    def predict_coarse(self, testing_data, fine2coarse, results_file):
        x_test, y_test = testing_data
        yc_test = tf.tensordot(y_test, fine2coarse, 1)
        x_test, yc_test = tf.cast(x_test, tf.dtypes.float32), tf.cast(yc_test, tf.dtypes.float32)

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
        x_test, yc_test = tf.cast(x_test, tf.dtypes.float32), tf.cast(y_test, tf.dtypes.float32)

        features_test = self.attention(x_test)

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

    def attention(self, x):

        # attention and cropping
        feature_model = tf.keras.models.Model(inputs=self.cc.input,
                                              outputs=self.cc.get_layer('conv2_block3_out').output)
        feature_map = feature_model(x)

        weights = tf.reduce_sum(feature_map, axis=(1, 2))
        weights = tf.math.l2_normalize(weights, axis=1)
        weights = tf.expand_dims(weights, axis=1)
        weights = tf.expand_dims(weights, axis=1)
        weigthed_channels = tf.multiply(feature_map, weights)
        attention_map = tf.expand_dims(tf.reduce_sum(weigthed_channels, 3), 3)
        cropped_features = tf.multiply(feature_map, attention_map)
        return cropped_features
