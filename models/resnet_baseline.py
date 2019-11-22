import datetime
import logging

import numpy as np
import tensorflow as tf

import models.plugins as plugins
import utils

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

        self.ckpt = None
        self.manager = None

        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        train_log_dir = self.logs_directory + '/' + current_time + '/train'
        test_log_dir = self.logs_directory + '/' + current_time + '/test'
        self.train_summary_writer = tf.summary.create_file_writer(train_log_dir)
        self.test_summary_writer = tf.summary.create_file_writer(test_log_dir)

        self.training_params = {
            'batch_size': 64,
            'initial_epoch': 0,
            'step': 5,  # Save weights every this amount of epochs
            'stop': 500,
            'lr': 0.001,
            'lr_decay': 1e-6,
            'fine_tune_epochs': 50
        }

        self.prediction_params = {
            'batch_size': 64
        }

    def train(self, training_data, validation_data):
        x_train, y_train = training_data
        x_val, y_val = validation_data

        p = self.training_params

        self.adam_coarse = tf.keras.optimizers.Adam(lr=p['lr'], decay=p['lr_decay'])
        self.loss_fun = tf.keras.losses.CategoricalCrossentropy()

        self.ckpt = tf.train.Checkpoint(step=tf.Variable(1), optimizer=self.adam_coarse, net=self.full_classifier)
        self.manager = tf.train.CheckpointManager(self.ckpt, self.logs_directory, max_to_keep=3)

        index = p['initial_epoch']

        # Unfreeze complete ResNet
        utils.unfreeze_layers(self.full_classifier.layers)

        logger.info('Training coarse stage')
        while index < p["fine_tune_epochs"]:
            for i in range(index, index + p["step"]):
                tr_loss = 0
                training_data = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(
                    batch_size=p["batch_size"])
                validation_data = tf.data.Dataset.from_tensor_slices((x_val, y_val)).batch(batch_size=p["batch_size"])
                for x, y in training_data:
                    tr_loss += self.train_step_coarse(x, y)
            val_loss, val_acc = self.evaluate(validation_data, len(y_val))
            with self.train_summary_writer.as_default():
                tf.summary.scalar('training loss', tr_loss, step=index)
                tf.summary.scalar('validation loss', val_loss, step=index)
                tf.summary.scalar('validation accuracy', val_acc, step=index)
            self.ckpt.step.assign_add(p['step'])
            self.manager.save()
            index += p['step']
            print(f"Epoch: {index}", f"Training loss: {tr_loss}", f"Validation loss: {val_loss}",
                  f"Validation loss: {val_acc}")

        # Freeze last layers of ResNet for tuning last layer
        utils.freeze_layers(self.full_classifier.layers[:-2])

        # Recompile model
        self.adam_fine = tf.keras.optimizers.Adam(lr=0.001, decay=1e-6)

        # Main train
        logger.info('Training fine stage')
        self.ckpt = tf.train.Checkpoint(step=tf.Variable(1), optimizer=self.adam_fine, net=self.full_classifier)
        while index < p['stop']:
            for i in range(index, index + p['step']):
                tr_loss = 0
                training_data = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(
                    batch_size=p["batch_size"])
                validation_data = tf.data.Dataset.from_tensor_slices((x_val, y_val)).batch(batch_size=p["batch_size"])
                for x, y in training_data:
                    tr_loss += self.train_step_fine(x, y) * p["batch_size"]
                tr_loss = tr_loss / len(y_train)
            val_loss, val_acc = self.evaluate(validation_data, len(y_val))
            with self.train_summary_writer.as_default():
                tf.summary.scalar('training loss', tr_loss, step=index)
                tf.summary.scalar('validation loss', val_loss, step=index)
                tf.summary.scalar('validation accuracy', val_acc, step=index)
            self.ckpt.step.assign_add(p['step'])
            self.manager.save()
            index += p['step']
            print(f"Epoch: {index}", f"Training loss: {tr_loss}", f"Validation loss: {val_loss}",
                  f"Validation loss: {val_acc}")

        # Unfreeze ResNet
        utils.unfreeze_layers(self.full_classifier.layers[:-2])

    def predict_fine(self, testing_data, results_file):
        x_test, y_test = testing_data

        p = self.prediction_params

        yh_s = self.full_classifier.predict(x_test, batch_size=p['batch_size'])

        single_classifier_error = utils.get_error(y_test, yh_s)
        logger.info('Single Classifier Error: ' + str(single_classifier_error))

        results_dict = {'Single Classifier Error': single_classifier_error}
        utils.write_results(results_file, results_dict=results_dict)

        return yh_s

    def predict_coarse(self, testing_data, results_file, fine2coarse):
        x_test, y_test = testing_data

        p = self.prediction_params

        yh_s = self.full_classifier.predict(x_test, batch_size=p['batch_size'])

        single_classifier_error = utils.get_error(y_test, yh_s)
        logger.info('Single Classifier Error: ' + str(single_classifier_error))

        yh_c = np.dot(yh_s, fine2coarse)
        y_test_c = np.dot(y_test, fine2coarse)
        coarse_classifier_error = utils.get_error(y_test_c, yh_c)

        logger.info('Single Classifier Error: ' + str(coarse_classifier_error))
        results_dict = {'Single Classifier Error': single_classifier_error,
                        'Coarse Classifier Error': coarse_classifier_error}
        utils.write_results(results_file, results_dict=results_dict)

    def build_full_classifier(self):

        model = tf.keras.applications.resnet.ResNet50(include_top=False, weights='imagenet',
                                                      input_tensor=None, input_shape=self.input_shape,
                                                      pooling=None, classes=1000)
        net = tf.keras.layers.Flatten()(model.output)
        net = tf.keras.layers.Dense(
            self.n_fine_categories, activation='softmax')(net)
        return tf.keras.models.Model(inputs=model.input, outputs=net)

    @tf.function
    def train_step_fine(self, x_train, y_train):
        with tf.GradientTape() as tape:
            tr_loss = self.loss_fun(self.full_classifier(x_train), y_train)
        gradients = tape.gradient(tr_loss, self.full_classifier.trainable_variables)
        self.adam_fine.apply_gradients(zip(gradients, self.full_classifier.trainable_variables))
        return tr_loss

    @tf.function
    def train_step_coarse(self, x_train, y_train):
        with tf.GradientTape() as tape:
            tr_loss = self.loss_fun(self.full_classifier(x_train), y_train)
        gradients = tape.gradient(tr_loss, self.full_classifier.trainable_variables)
        self.adam_coarse.apply_gradients(zip(gradients, self.full_classifier.trainable_variables))
        return tr_loss

    def evaluate(self, validation_data, n):
        val_acc = 0
        val_loss = 0
        for x, y in validation_data:
            y_pred = self.full_classifier(x)
            val_loss += self.loss_fun(y, y_pred) * len(y)
            tmp = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y, 1))
            val_acc += tf.reduce_sum(tmp, tf.float32)
        val_acc = val_acc / n
        return val_loss, val_acc
