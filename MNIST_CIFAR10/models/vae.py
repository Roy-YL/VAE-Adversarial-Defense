from keras.layers import Flatten, Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Lambda, Reshape, Conv2DTranspose, \
    AveragePooling2D
from keras.models import Model
from keras import backend as K
from keras.losses import binary_crossentropy, mean_squared_error
import numpy as np

def sampling(args):
    z_mean, z_log_sigma = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    # by default, random_normal has mean=0 and std=1.0
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_sigma) * epsilon


def vae_model_mnist(latent_dim=256):
    inputs = Input(shape=(28, 28, 1))  #
    latent_dim = latent_dim

    x = Conv2D(16, (3, 3), activation='relu', padding='same')(inputs)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    if latent_dim == 128:
        x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
        x = MaxPooling2D((2, 2), padding='same')(x)

    conv_shape = x.get_shape().as_list()[1:]
    conv_dim = int(conv_shape[0]) * int(conv_shape[1]) * int(conv_shape[2])

    x = Flatten()(x)
    z_mean = Dense(latent_dim)(x)
    z_log_sigma = Dense(latent_dim)(x)

    z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_sigma])
    encoder = Model(inputs, [z_mean, z_log_sigma, z], name='encoder')
    print(encoder.summary())

    latent_inputs = Input(shape=(latent_dim,))
    x = Dense(conv_dim)(latent_inputs)
    x = Reshape(conv_shape)(x)

    if latent_dim == 128:
        x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
        x = UpSampling2D((2, 2))(x)

    x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    if latent_dim == 128:
        x = Conv2D(16, (3, 3), activation='relu')(x)
    else:
        x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

    decoder = Model(latent_inputs, decoded, name='decoder')
    print(decoder.summary())
    outputs = decoder(encoder(inputs)[2])
    vae = Model(inputs, outputs, name='VAE')

    reconstruction_loss = mean_squared_error(K.flatten(inputs), K.flatten(outputs))
    reconstruction_loss *= 784
    kl_loss = 1 + z_log_sigma - K.square(z_mean) - K.exp(z_log_sigma)
    kl_loss = K.sum(kl_loss, axis=-1)
    kl_loss *= -0.5
    vae_loss = K.mean(reconstruction_loss + kl_loss * 0.01)
    vae.add_loss(vae_loss)

    return vae


def vae_model_cifar(latent_dim=1024):
    inputs = Input(shape=(32, 32, 3))
    latent_dim = latent_dim

    x = Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)

    conv_shape = x.get_shape().as_list()[1:]
    conv_dim = int(conv_shape[0]) * int(conv_shape[1]) * int(conv_shape[2])

    print(conv_shape, conv_dim)

    x = Flatten()(x)
    z_mean = Dense(latent_dim)(x)
    z_log_sigma = Dense(latent_dim)(x)

    z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_sigma])
    encoder = Model(inputs, [z_mean, z_log_sigma, z], name='encoder')
    print(encoder.summary())

    latent_inputs = Input(shape=(latent_dim,))
    x = Dense(conv_dim)(latent_inputs)
    x = Reshape(conv_shape)(x)

    x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    decoded = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)

    decoder = Model(latent_inputs, decoded, name='decoder')
    # print(decoder.summary())
    outputs = decoder(encoder(inputs)[2])
    vae = Model(inputs, outputs, name='VAE')

    reconstruction_loss = mean_squared_error(K.flatten(inputs), K.flatten(outputs))
    reconstruction_loss *= 3072
    kl_loss = 1 + z_log_sigma - K.square(z_mean) - K.exp(z_log_sigma)
    kl_loss = K.sum(kl_loss, axis=-1)
    kl_loss *= -0.5

    vae_loss = K.mean(reconstruction_loss + kl_loss * 0.01)
    vae.add_loss(vae_loss)

    return vae

# vae_model_mnist().summary()
# vae_model_cifar().summary()
