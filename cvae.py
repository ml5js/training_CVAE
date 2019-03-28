import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.datasets import mnist
from keras.layers import Input, Dense, Lambda, Conv2D, Conv2DTranspose, MaxPooling2D, UpSampling2D, Reshape, Flatten
from keras.layers.merge import concatenate as concat
from keras.models import Model
from keras.optimizers import Adam


# class for CVAE
class CVAE():
    def __init__(self, args):
        # hyperparamters
        self.args = args
        self.n_dim = 2
        self.filters = 8
        self.epochs = 50
        self.batch_size = 100
    
    # sample z
    def sample_z(self, args):
        mu, std = args
        eps = K.random_normal(shape=(self.batch_size, self.n_dim), mean=0., stddev=1.)
        return mu + K.exp(std / 2) * eps

    # loss function
    def vae_loss(self, y_true, y_pred):
        recon = 28*28*K.binary_crossentropy(y_true, y_pred)
        kl = 0.5 * K.sum(K.exp(self.std) + K.square(self.mu) - 1. - self.std, axis=-1)
        return recon + kl

    # encoder
    def encode(self, X, label):
        x_label = Dense(28 * 28)(label)
        x_label = Reshape((28, 28, 1))(x_label)
        x = concat([X, x_label])

        for i in range(2):
            x = Conv2D(filters=self.filters, kernel_size=(3,3), activation='relu', padding='same')(x)
            x = MaxPooling2D((2, 2), padding='same')(x)
            self.filters *= 2

        shape = K.int_shape(x)

        x = Flatten()(x)
        x = Dense(16, activation='relu')(x)
        self.mu = Dense(self.n_dim, activation='linear')(x)
        self.std = Dense(self.n_dim, activation='linear')(x)

        z = Lambda(self.sample_z, output_shape=(self.n_dim, ))([self.mu, self.std])
        encoder = Model([X, label], [self.mu, self.std, z], name='encoder')
        
        return encoder, shape

    
    # decoder
    def decode(self, z_inputs, label, shape):
        x = concat([z_inputs, label])
        x = Dense(shape[1]*shape[2]*shape[3], activation='relu')(x)
        x = Reshape((shape[1], shape[2], shape[3]))(x)

        for i in range(2):
            x = Conv2DTranspose(filters=self.filters, kernel_size=(3,3), activation='relu', padding='same')(x)
            x = UpSampling2D((2, 2))(x)
            self.filters //= 2

        outputs = Conv2DTranspose(filters=1, kernel_size=(3,3), activation='sigmoid', padding='same')(x)
        decoder = Model([z_inputs, label], outputs, name='decoder')

        return decoder

    # forward
    def forward(self, X_train, X_test, y_train, y_test):
        X_shape = X_train.shape[1]
        y_shape = y_train.shape[1]
        X = Input(shape=(28, 28, 1), name='input')
        label = Input(shape=(y_shape,), name='label')
        
        encoder, shape = self.encode(X, label)
        encoder.summary()

        z_inputs = Input(shape=(self.n_dim,), name='latent_input')
        decoder = self.decode(z_inputs, label, shape)
        decoder.summary()

        z_output = encoder([X, label])[2]
        outputs = decoder([z_output, label])
        cvae = Model([X, label], outputs, name='cvae')
        cvae.compile(optimizer=Adam(lr=0.0005), loss=self.vae_loss)
        cvae.summary()
        cvae_hist = cvae.fit([X_train, y_train], X_train, verbose=1, batch_size=self.batch_size, epochs=self.epochs,
                     validation_data=([X_test, y_test], X_test))
        
        return cvae, cvae_hist


