import os
import numpy as np
import pandas as pd
from PIL import Image, ImageFilter
from scipy.misc import imread
import imageio
import tensorflow as tf
from tensorflow.contrib.slim.nets import inception
import scipy.io as sio
from image_utils import load_images, load_imagefiles, jpeg, l2difference, sharpen, smoothen
from model_utils import InceptionModel, print_accuracy, reconstruct_image
from patch_vae import patch_vae_model
from vae_clip import patch_vae_model_64
from patch_vae_16 import patch_vae_model_16
from io import BytesIO
from sklearn.feature_extraction.image import extract_patches_2d
import keras
from cleverhans.attacks import FastGradientMethod
import csv
from keras import backend as K
import matplotlib.pyplot as plt

plt.switch_backend('agg')

slim = tf.contrib.slim
checkpoint_path = './inception-v3/inception_v3.ckpt'
batch_shape = [1000, 299, 299, 3]
num_classes = 1001
image_labels = pd.read_csv("./image_info/images.csv")
attack_epsilon = 0.005
patch_size = 32

keras.backend.clear_session()
tf.reset_default_graph()

images = next(load_imagefiles('./images/', batch_shape))
# non_targeted_images = next(load_imagefiles('./nontargeted_images_{}'.format(attack_epsilon), batch_shape))
originalimages = (images[1] + 1.0) / 2.0
filenames = images[0]
true_labels = []
for i in range(len(filenames)):
    true_labels.append(int(image_labels[image_labels['ImageId'] == filenames[i][:-4]]['TrueLabel']))

keras.backend.clear_session()
tf.reset_default_graph()

sess = tf.Session()
keras.backend.set_session(sess)

vae_model_16, encoder_16, decoder_16, reconstruction_loss_16, kl_loss_16, inputs_16, latent_16, outputs_16 = patch_vae_model(
    input_size=16)
vae_model, encoder, decoder, reconstruction_loss, kl_loss, inputs, latent, outputs = patch_vae_model()
vae_model_64, encoder_64, decoder_64, reconstruction_loss_64, kl_loss_64, inputs_64, latent_64, outputs_64 = patch_vae_model(
    input_size=64)

AdamOptimizer = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False,
                                      clipvalue=0.01, clipnorm=0.5)
vae_model.add_loss(reconstruction_loss)
vae_model.add_loss(kl_loss)
vae_model.compile(optimizer=AdamOptimizer)
vae_model.load_weights('vae_32.h5')

vae_model_64.add_loss(reconstruction_loss_64)
vae_model_64.add_loss(kl_loss_64)
vae_model_64.compile(optimizer=AdamOptimizer)
vae_model_64.load_weights('vae_64.h5')

vae_model_16.add_loss(reconstruction_loss_16)
vae_model_16.add_loss(kl_loss_16)
vae_model_16.compile(optimizer=AdamOptimizer)
vae_model_16.load_weights('vae_16.h5')

# vae_images = reconstruct_image(images=((nontargeted_images+1.0)/2.0), model=vae_model,patch_size=64,step_size=64)

epses = [0.005, 0.0135, 0.022, 0.0305, 0.039, 0.0475, 0.056, 0.0645,
         0.073, 0.0815, 0.09]
results = []
for i in epses:
    # nontargeted_images = next(load_images('./nontargeted_images_{}'.format(i), batch_shape, is_training=False))
    nontargeted_images = next(
        load_images('/work/yl490/adversarial_images/nontargeted_images_{}'.format(i), batch_shape, is_training=False))

    vae_images_1608 = reconstruct_image(images=((nontargeted_images + 1.0) / 2.0), model=vae_model_16, patch_size=16,
                                        step_size=8)
    vae_images_1616 = reconstruct_image(images=((nontargeted_images + 1.0) / 2.0), model=vae_model_16, patch_size=16,
                                        step_size=16)

    vae_images_3216 = reconstruct_image(images=((nontargeted_images + 1.0) / 2.0), model=vae_model, patch_size=32,
                                        step_size=16)
    vae_images_3232 = reconstruct_image(images=((nontargeted_images + 1.0) / 2.0), model=vae_model,
                                        patch_size=patch_size, step_size=32)
    vae_images_6432 = reconstruct_image(images=((nontargeted_images + 1.0) / 2.0), model=vae_model_64, patch_size=64,
                                        step_size=32)
    vae_images_6464 = reconstruct_image(images=((nontargeted_images + 1.0) / 2.0), model=vae_model_64, patch_size=64,
                                        step_size=64)
    '''
    vae_images_1608 = sharpen(vae_images_1608)
    vae_images_1616 = sharpen(vae_images_1616)
    vae_images_3216 = sharpen(vae_images_3216)
    vae_images_3232 = sharpen(vae_images_3232)
    vae_images_6432 = sharpen(vae_images_6432)
    vae_images_6464 = sharpen(vae_images_6464)
    '''
    vae_images_1608 = smoothen(vae_images_1608)
    vae_images_1616 = smoothen(vae_images_1616)
    vae_images_3216 = smoothen(vae_images_3216)
    vae_images_3232 = smoothen(vae_images_3232)
    vae_images_6432 = smoothen(vae_images_6432)
    vae_images_6464 = smoothen(vae_images_6464)

    mix1 = (vae_images_1608 + vae_images_3216 + vae_images_6432) / 3
    mix2 = (vae_images_1616 + vae_images_3232 + vae_images_6464) / 3
    mix3 = (
                       vae_images_1616 + vae_images_3232 + vae_images_6464 + vae_images_1608 + vae_images_3216 + vae_images_6432) / 6

    result = [i,
              l2difference((originalimages * 2.0 - 1.0), nontargeted_images),
              print_accuracy(nontargeted_images, true_labels, jpeg_reconstruct=False),
              print_accuracy(nontargeted_images, true_labels, jpeg_reconstruct=True, jpeg_quality=10),
              print_accuracy(nontargeted_images, true_labels, jpeg_reconstruct=True),
              print_accuracy(nontargeted_images, true_labels, jpeg_reconstruct=True, jpeg_quality=50),
              print_accuracy(nontargeted_images, true_labels, jpeg_reconstruct=True, jpeg_quality=70),
              print_accuracy(vae_images_1608 * 2.0 - 1.0, true_labels),
              print_accuracy(vae_images_1616 * 2.0 - 1.0, true_labels),
              print_accuracy(vae_images_3216 * 2.0 - 1.0, true_labels),
              print_accuracy(vae_images_3232 * 2.0 - 1.0, true_labels),
              print_accuracy(vae_images_6432 * 2.0 - 1.0, true_labels),
              print_accuracy(vae_images_6464 * 2.0 - 1.0, true_labels),
              print_accuracy(mix1 * 2.0 - 1.0, true_labels),
              print_accuracy(mix2 * 2.0 - 1.0, true_labels),
              print_accuracy(mix3 * 2.0 - 1.0, true_labels)]
    print(result)
    results.append(result)
    with open('fgsm_results_163264.csv', 'a') as csvfile:
        # print("writing {}".format())
        writer = csv.writer(csvfile)
        writer.writerow(result)
        csvfile.close()
