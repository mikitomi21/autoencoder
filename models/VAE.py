import pickle

from keras.layers import Input, Dense, Flatten, Conv2D, BatchNormalization, Dropout, LeakyReLU, Reshape, \
    Conv2DTranspose, Activation
from keras import backend
from keras.models import Model
from keras.optimizers import Adam
from keras.utils import plot_model

import numpy as np
import os


class VAE:
    def __init__(
            self,
            input_dim,
            encoder_filters,
            encoder_kernel_size,
            encoder_strides,
            decoder_filters,
            decoder_kernel_size,
            decoder_strides,
            z_dim):
        self.input_dim = input_dim
        self.encoder_filters = encoder_filters
        self.encoder_kernel_size = encoder_kernel_size
        self.encoder_strides = encoder_strides
        self.decoder_filters = decoder_filters
        self.decoder_kernel_size = decoder_kernel_size
        self.decoder_strides = decoder_strides
        self.number_of_conv_layers = self.get_number_of_conv_layers()
        self.z_dim = z_dim

        self._build()

    def get_number_of_conv_layers(self):
        if len(self.encoder_filters) == len(self.decoder_filters):
            return len(self.encoder_filters)
        raise ValueError("The length of encoder_filters and decoder_filters are not the same")

    def _build(self):
        # encoder
        encoder_input = Input(shape=self.input_dim, name='encoder_input')
        x = encoder_input

        for i in range(self.number_of_conv_layers):
            conv_layer = Conv2D(
                filters=self.encoder_filters[i],
                kernel_size=self.encoder_kernel_size[i],
                strides=self.encoder_strides[i],
                padding="same",
                name=f"conv_layer{i}"
            )
            x = conv_layer(x)

            x = BatchNormalization()(x)

            x = LeakyReLU()(x)

            x = Dropout(rate=0.25)(x)

        shape_before_flattening = backend.int_shape(x)[1:]

        x = Flatten()(x)

        encoder_output = Dense(self.z_dim, name='encoder_output')(x)

        self.encoder = Model(encoder_input, encoder_output)

        # decoder
        decoder_input = Input(shape=(self.z_dim,), name='decoder_input')
        x = Dense(np.prod(shape_before_flattening))(decoder_input)

        x = Reshape(shape_before_flattening)(x)

        for i in range(self.number_of_conv_layers):
            conv_t_layer = Conv2DTranspose(
                filters=self.decoder_filters[i],
                kernel_size=self.decoder_kernel_size[i],
                strides=self.decoder_strides[i],
                padding="same",
                name=f"conv_layer{i}"
            )

            if i < self.number_of_conv_layers - 1:
                x = conv_t_layer(x)

                x = BatchNormalization()(x)

                x = LeakyReLU()(x)

                x = Dropout(rate=0.25)(x)
            else:
                x = conv_t_layer(x)
                x = Activation('sigmoid')(x)

        decoder_output = x

        self.decoder = Model(decoder_input, decoder_output)

        # full autoencoder
        model_input = encoder_input
        model_output = self.decoder(encoder_output)
        self.model = Model(model_input, model_output)

    def compile(self, learning_rate):
        optimizer = Adam(learning_rate=learning_rate)

        def loss_fun(y_real, y_pred):
            return backend.mean(backend.square(y_pred - y_real), axis=[1, 2, 3])

        self.model.compile(optimizer=optimizer, loss=loss_fun, metrics=['accuracy'])

    def save(self, folder):
        if not os.path.exists(folder):
            os.makedirs(folder)
            os.makedirs(os.path.join(folder, 'weights'))
            os.makedirs(os.path.join(folder, 'viz'))
            os.makedirs(os.path.join(folder, 'images'))

        with open(os.path.join(folder, 'params.pkl'), 'wb') as f:
            pickle.dump([
                self.input_dim,
                self.encoder_filters,
                self.encoder_kernel_size,
                self.encoder_strides,
                self.decoder_filters,
                self.decoder_kernel_size,
                self.decoder_strides,
                self.number_of_conv_layers,
                self.z_dim,
            ], f)

        self.plot_model(folder)

    def load_weights(self, file_path):
        self.model.load_weights(file_path)

    def train(self, x_train, batch_size, epochs, shuffle):
        self.model.fit(
            x_train,
            batch_size=batch_size,
            epochs=epochs,
            shuffle=shuffle,
        )

    def plot_model(self, folder):
        plot_model(self.model, to_file=os.path.join(folder, 'viz/model.png'), show_shapes=True, show_layer_names=True)
        plot_model(self.encoder, to_file=os.path.join(folder, 'viz/encoder.png'), show_shapes=True, show_layer_names=True)
        plot_model(self.decoder, to_file=os.path.join(folder, 'viz/decoder.png'), show_shapes=True, show_layer_names=True)
