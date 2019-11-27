import datetime

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

        self.cc, self.fc = None, None
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

        self.tbCallback_train = tf.keras.callbacks.TensorBoard(
            log_dir=logs_directory + '/' + current_time,
            update_freq='epoch')  # How often to write logs (default: once per epoch)
        self.early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
        self.reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                                                              patience=5, min_lr=0.0000001)
        self.model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
            filepath=model_directory + "resnet_attention_{epoch:02d}-epochs.h5",
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            save_freq=90000)

        self.training_params = {
            'batch_size': 64,
            'initial_epoch': 0,
            'lr_coarse': 3e-5,
            'lr_fine': 1e-5,
            'step': 5,  # Save weights every this amount of epochs
            'stop': 100
        }

        self.prediction_params = {
            'batch_size': 64
        }

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

        self.cc.fit(x_train, yc_train,
                    batch_size=p['batch_size'],
                    initial_epoch=index,
                    epochs=index + p['stop'],
                    validation_data=(x_val, yc_val),
                    callbacks=[self.tbCallback_train, self.early_stopping,
                               self.reduce_lr, self.model_checkpoint])

        # logger.info('Predicting Coarse Labels')
        # yc_pred = self.cc.predict(x_train, batch_size=p["batch_size"])
        # yc_val_pred = self.cc.predict(x_val, batch_size=p["batch_size"])

        # logger.info('Saving Coarse Labels')
        # np.save(self.model_directory + "/yc_pred", yc_pred)
        # np.save(self.model_directory + "/yc_val_pred", yc_val_pred)

        # logger.info('Clearing Coarse Training Session')
        # tf.keras.backend.clear_session()

    def train_fine(self, training_data, validation_data, fine2coarse):
        x_train, y_train = training_data
        yc_train = tf.linalg.matmul(y_train, fine2coarse)
        x_val, y_val = validation_data
        yc_val = tf.linalg.matmul(y_val, fine2coarse)

        # logger.info('Loading Coarse Predictions')
        # yc_pred = tf.convert_to_tensor(np.load(self.model_directory + "/yc_pred.npy"))
        # yc_val_pred = tf.convert_to_tensor(np.load(self.model_directory + "/yc_val_pred.npy"))

        p = self.training_params

        logger.debug(f"Creating fine classifier with shared layers")
        self.cc, self.fc = self.build_cc_fc()

        logger.info("Attention reweighting of training features")
        feature_map_att = self.attention(x_train)
        logger.info("Attention reweighting of validation features")
        feature_map_att_val = self.attention(x_val)

        logger.info('Start Fine Classification Training')

        adam_fine = tf.keras.optimizers.Adam(lr=p['lr_fine'])
        self.fc.compile(optimizer=adam_fine,
                        loss='categorical_crossentropy',
                        metrics=['accuracy'])
        index = p['initial_epoch']

        self.fc.fit([feature_map_att, yc_train], y_train,
                    batch_size=p['batch_size'],
                    initial_epoch=index,
                    epochs=index + p['stop'],
                    validation_data=([feature_map_att_val, yc_val], y_val),
                    callbacks=[self.tbCallback_train, self.early_stopping,
                               self.reduce_lr, self.model_checkpoint])

        # logger.info('Clearing Fine Training Session')
        # tf.keras.backend.clear_session()

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

    def attention(self, x, batch_size=32):

        # attention and cropping
        feature_model = tf.keras.models.Model(inputs=self.cc.input,
                                              outputs=self.cc.get_layer('conv2_block3_out').output)
        feature_map = feature_model.predict(x, batch_size=batch_size)

        cropped_features = list()
        ds = tf.data.Dataset.from_tensor_slices(feature_map).batch(batch_size)
        for x in ds:
            weights = tf.reduce_sum(x, axis=(1, 2))
            weights = tf.math.l2_normalize(weights, axis=1)
            weights = tf.expand_dims(weights, axis=1)
            weights = tf.expand_dims(weights, axis=1)
            weigthed_channels = tf.multiply(x, weights)
            attention_map = tf.expand_dims(tf.reduce_sum(weigthed_channels, 3), 3)
            cropped_features.append(tf.multiply(x, attention_map))
        cropped_features = tf.concat(cropped_features, axis=0)
        return cropped_features
