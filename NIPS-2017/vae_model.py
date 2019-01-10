from keras.layers import Flatten, Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Lambda, Reshape, Conv2DTranspose, \
    Cropping2D, AveragePooling2D, Lambda
from keras.models import Model
from keras import backend as K
from keras.losses import binary_crossentropy, mean_squared_error
import numpy as np


def sampling(args, clip_low=-5, clip_high=5):
    z_mean, z_log_sigma = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    # by default, random_normal has mean=0 and std=1.0
    epsilon = K.random_normal(shape=(batch, dim))
    epsilon = K.clip(epsilon, clip_low, clip_high)
    return z_mean + K.exp(0.5 * z_log_sigma) * epsilon


def patch_vae_model(input_size=64, latent_dim=1024):
    inputs = Input(shape=(input_size, input_size, 3))
    latent_dim = latent_dim

    x = Conv2D(128, (3, 3), activation='relu', padding='same')(inputs)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)

    conv_shape = x.get_shape().as_list()[1:]
    conv_dim = int(conv_shape[0]) * int(conv_shape[1]) * int(conv_shape[2])

    x = Flatten()(x)
    z_mean = Dense(latent_dim)(x)
    z_log_sigma = Dense(latent_dim)(x)

    z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_sigma])
    encoder = Model(inputs, [z_mean, z_log_sigma, z], name='encoder')
    # print(encoder.summary())

    latent_inputs = Input(shape=(latent_dim,))
    x = Dense(conv_dim)(latent_inputs)
    x = Reshape(conv_shape)(x)

    x = Conv2DTranspose(32, (3, 3), activation='relu', padding='same')(x)
    x = Conv2DTranspose(64, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2DTranspose(128, (3, 3), activation='relu', padding='same')(x)
    decoded = Conv2DTranspose(3, (3, 3), activation='sigmoid', padding='same')(x)
    decoded = Lambda(lambda x: K.clip(x, 0., 1.))(decoded)

    decoder = Model(latent_inputs, decoded, name='decoder')
    # print(decoder.summary())
    outputs = decoder(encoder(inputs)[2])
    vae = Model(inputs, outputs, name='VAE')

    reconstruction_loss = binary_crossentropy(inputs, outputs)
    reconstruction_loss *= (input_size * input_size * 3)

    kl_loss = - 0.5 * K.mean(1 + z_log_sigma - K.square(z_mean) - K.exp(z_log_sigma), axis=-1)

    return vae, encoder, decoder, reconstruction_loss, kl_loss, inputs, z, outputs


#vae, encoder, decoder, reconstruction_loss, kl_loss, inputs, z, outputs = patch_vae_model()
#vae.summary()
