import numpy as np
import tensorflow as tf
from keras.layers import Input, Dense, Lambda, Conv2D, Conv2DTranspose, MaxPooling2D, UpSampling2D, Reshape, Flatten
from keras.layers.merge import concatenate as concat
from keras.models import Model
from keras import backend as K
from keras.datasets import mnist
from keras.utils import to_categorical
from keras.optimizers import Adam

(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

X_train = np.reshape(X_train, [-1, 28, 28, 1])
X_test = np.reshape(X_test, [-1, 28, 28, 1])
X_train = X_train.astype('float32') / 255.
X_test = X_test.astype('float32') / 255.

y_train = to_categorical(Y_train)
y_test = to_categorical(Y_test)

X_shape = X_train.shape[1]
y_shape = y_train.shape[1]

# hyperparamters
n_dim = 2
filters = 8
epochs = 50
batch_size = 100

# decoder
X = Input(shape=(28, 28, 1), name='input')
label = Input(shape=(y_shape,), name='label')
x_label = Dense(28 * 28)(label)
x_label = Reshape((28, 28, 1))(x_label)
x = concat([X, x_label])

for i in range(2):
    x = Conv2D(filters=filters, kernel_size=(3,3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    filters *= 2

shape = K.int_shape(x)
print(shape)
x = Flatten()(x)
x = Dense(16, activation='relu')(x)
mu = Dense(n_dim, activation='linear')(x)
std = Dense(n_dim, activation='linear')(x)

# sample z
def sample_z(args):
    mu, std = args
    eps = K.random_normal(shape=(batch_size, n_dim), mean=0., stddev=1.)
    return mu + K.exp(std / 2) * eps

z = Lambda(sample_z, output_shape=(n_dim, ))([mu, std])

encoder = Model([X, label], [mu, std, z], name='encoder')
encoder.summary()

# decoder
print(z.shape)
z_inputs = Input(shape=(n_dim,), name='latent_input')
x = concat([z_inputs, label])
x = Dense(shape[1]*shape[2]*shape[3], activation='relu')(x)
x = Reshape((shape[1], shape[2], shape[3]))(x)

for i in range(2):
    x = Conv2DTranspose(filters=filters, kernel_size=(3,3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    filters //= 2

outputs = Conv2DTranspose(filters=1, kernel_size=(3,3), activation='sigmoid', padding='same')(x)

decoder = Model([z_inputs, label], outputs, name='decoder')
decoder.summary()

# loss function
def vae_loss(y_true, y_pred):
    recon = 28*28*K.binary_crossentropy(y_true, y_pred)
    kl = 0.5 * K.sum(K.exp(std) + K.square(mu) - 1. - std, axis=-1)
    return recon + kl

z_output = encoder([X, label])[2]
outputs = decoder([z_output, label])
cvae = Model([X, label], outputs, name='cvae')
cvae.compile(optimizer=Adam(lr=0.0005), loss=vae_loss)
cvae.summary()
cvae_hist = cvae.fit([X_train, y_train], X_train, verbose=1, batch_size=batch_size, epochs=epochs,
                     validation_data=([X_test, y_test], X_test))

# constructing model
# encoder = Model([X, label], mu)

# d_in = Input(shape=(n_dim + y_shape,))
# d_h = decoder_h(d_in)
# d_out = decoder_out(d_h)
# decoder = Model(d_in, d_out)

# decoder.save('decoder.h5')