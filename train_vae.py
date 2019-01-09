import os
from cleverhans.attacks import FastGradientMethod
import numpy as np
import pandas as pd
from PIL import Image
from scipy.misc import imread
import imageio
import tensorflow as tf
from tensorflow.contrib.slim.nets import inception
import scipy.io as sio
from image_utils import load_patches
from vae_clip import patch_vae_model
from sklearn.feature_extraction.image import extract_patches_2d
import keras

# set patch_size

patch_size = (16, 16)
model_name = "vae_{}".format(patch_size[0])
steps_per_epoch = 50  # for Keras fit_generator
epochs = 100
show_loss_on_samples = True  # Calculate and record loss on sampled image patches
image_dir = "../images"

sess = tf.Session()
keras.backend.set_session(sess)
vae_model, encoder, decoder, reconstruction_loss, kl_loss, inputs, z, outputs = patch_vae_model(
    input_size=patch_size[0])

AdamOptimizer = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

vae_model.add_loss(reconstruction_loss)
vae_model.add_loss(0.01 * kl_loss)
vae_model.compile(optimizer=AdamOptimizer)


class PrintLoss(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.recons_losses = []
        self.kl_losses = []

    def on_epoch_end(self, epoch, logs={}):
        self.recons_loss = 0
        self.kl_loss_ = 0
        for i in range(100):
            self.patches = next(load_patches('./images', patch_size=patch_size))[0]
            self.recons_loss += np.mean(sess.run(reconstruction_loss, feed_dict={inputs: self.patches}))
            self.kl_loss_ += np.mean(sess.run(kl_loss, feed_dict={inputs: self.patches}))
        self.recons_losses.append(self.recons_loss / 100)
        self.kl_losses.append(self.kl_loss_ / 100)


callback_list = []
if show_loss_on_samples:
    loss_callback = PrintLoss()
    callback_list.append(loss_callback)

try:
    print("Loading weights from " + model_name + ".h5")
    vae_model.load_weights(model_name + ".h5")
    print("Successfully loaded.")
except:
    print("Loading failure, starting to train model")
    vae_model.fit_generator(load_patches("./images", patch_size=patch_size), steps_per_epoch=50, epochs=1000,
                            verbose=2, callbacks=callback_list)

if show_loss_on_samples:
    print(loss_callback.recons_losses[-20:])
    print(loss_callback.kl_losses[-20:])

vae_model.save_weights(model_name + ".h5")
