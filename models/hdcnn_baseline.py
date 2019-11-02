import tensorflow as tf
import logging

logger = logging.getLogger('HDCNNBaseline')


class HDCNNBaseline:
    def __init__(self, logs_directory=None, model_directory=None, args=None):
        """
        HD-CNN baseline model

        """
        self.model_directory = model_directory
        self.args = args

        self.model = self.build_net()
        sgd_coarse = tf.keras.optimizers.SGD(
            lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        self.model.compile(optimizer=sgd_coarse,
                           loss='categorical_crossentropy',
                           metrics=['accuracy'])
        self.tbCallBack = tf.keras.callbacks.TensorBoard(
            log_dir=logs_directory, histogram_freq=0,
            write_graph=True, write_images=True)
        self.training_params = {
            'batch_size': 64,
            'initial_epoch': 0,
            'step': 5,  # Save weights every this amount of epochs
            'epochs': 30
        }

    def train(self, training_data, validation_data):
        x_train, y_train = training_data
        x_val, y_val = validation_data

        p = self.training_params

        index = 0
        while index < p['epochs']:
            self.model.fit(x_train, y_train,
                           batch_size=p['batch_size'],
                           initial_epoch=p['initial_epoch'],
                           epochs=p['initial_epoch'] + p['step'],
                           validation_data=(x_val, y_val),
                           callbacks=[self.tbCallBack])
            index += p['step']
            self.model.save_weights(self.model_directory + str(index))

    def predict(self, testing_data, results_directory):
        pred = None
        return pred

    def build_net(self):
        in_layer = tf.keras.Input(
            shape=(32, 32, 3), dtype='float32', name='main_input')

        net = tf.keras.layers.Conv2D(384, 3, strides=1, padding='same',
                                     activation='elu')(in_layer)
        net = tf.keras.layers.MaxPooling2D((2, 2), padding='valid')(net)

        net = tf.keras.layers.Conv2D(
            384, 1, strides=1, padding='same', activation='elu')(net)
        net = tf.keras.layers.Conv2D(
            384, 2, strides=1, padding='same', activation='elu')(net)
        net = tf.keras.layers.Conv2D(
            640, 2, strides=1, padding='same', activation='elu')(net)
        net = tf.keras.layers.Conv2D(
            640, 2, strides=1, padding='same', activation='elu')(net)
        net = tf.keras.layers.Dropout(.2)(net)
        net = tf.keras.layers.MaxPooling2D((2, 2), padding='valid')(net)

        net = tf.keras.layers.Conv2D(
            640, 1, strides=1, padding='same', activation='elu')(net)
        net = tf.keras.layers.Conv2D(
            768, 2, strides=1, padding='same', activation='elu')(net)
        net = tf.keras.layers.Conv2D(
            768, 2, strides=1, padding='same', activation='elu')(net)
        net = tf.keras.layers.Conv2D(
            768, 2, strides=1, padding='same', activation='elu')(net)
        net = tf.keras.layers.Dropout(.3)(net)
        net = tf.keras.layers.MaxPooling2D((2, 2), padding='valid')(net)

        net = tf.keras.layers.Conv2D(
            768, 1, strides=1, padding='same', activation='elu')(net)
        net = tf.keras.layers.Conv2D(
            896, 2, strides=1, padding='same', activation='elu')(net)
        net = tf.keras.layers.Conv2D(
            896, 2, strides=1, padding='same', activation='elu')(net)
        net = tf.keras.layers.Dropout(.4)(net)
        net = tf.keras.layers.MaxPooling2D((2, 2), padding='valid')(net)

        net = tf.keras.layers.Conv2D(
            896, 3, strides=1, padding='same', activation='elu')(net)
        net = tf.keras.layers.Conv2D(
            1024, 2, strides=1, padding='same', activation='elu')(net)
        net = tf.keras.layers.Conv2D(
            1024, 2, strides=1, padding='same', activation='elu')(net)
        net = tf.keras.layers.Dropout(.5)(net)
        net = tf.keras.layers.MaxPooling2D((2, 2), padding='valid')(net)

        net = tf.keras.layers.Conv2D(
            1024, 1, strides=1, padding='same', activation='elu')(net)
        net = tf.keras.layers.Conv2D(
            1152, 2, strides=1, padding='same', activation='elu')(net)
        net = tf.keras.layers.Dropout(.6)(net)
        net = tf.keras.layers.MaxPooling2D((2, 2), padding='same')(net)

        net = tf.keras.layers.Flatten()(net)
        net = tf.keras.layers.Dense(1152, activation='elu')(net)
        net = tf.keras.layers.Dense(100, activation='softmax')(net)
        return tf.keras.models.Model(inputs=in_layer, outputs=net)

    def load_weights(self, weights_file):
        self.model.load_weights(weights_file)
