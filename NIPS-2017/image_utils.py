import os
from io import BytesIO
import numpy as np
import pandas as pd
from PIL import Image, ImageFilter
from scipy.misc import imread
from scipy.misc import imsave
import tensorflow as tf
from tensorflow.contrib.slim.nets import inception
from sklearn.feature_extraction.image import extract_patches_2d
import imageio

slim = tf.contrib.slim

def jpeg(X, image_size=299, channels=3, quality=23):
    X_jpeg = np.zeros(X.shape)
    X = X.reshape((-1, image_size, image_size, channels))
    if channels == 1:
        mode = 'L'
        X = X.reshape((-1, image_size, image_size))
    elif channels == 3:
        mode = 'RGB'

    for i in range(len(X)):
        f = BytesIO()
        im = np.uint8(X[i, :] * 255)
        Image.fromarray(im, mode=mode).save(f, "jpeg", quality=quality)
        im_jpeg = Image.open(f)
        im_jpeg = np.array(im_jpeg.getdata())
        im_jpeg = im_jpeg.reshape((image_size, image_size, channels))
        X_jpeg[i, :] = im_jpeg / 255
    return X_jpeg


def load_images(input_dir, batch_shape, is_training=False):
    while True:
        images = np.zeros(batch_shape)
        filenames = []
        idx = 0
        batch_size = batch_shape[0]
        for filepath in tf.gfile.Glob(os.path.join(input_dir, '*.png')):
            with tf.gfile.Open(filepath, "rb") as f:
                image = imread(f, mode='RGB').astype(np.float) / 255.0
            # Images for inception classifier are normalized to be in [-1, 1] interval.
            images[idx, :, :, :] = image * 2.0 - 1.0
            filenames.append(os.path.basename(filepath))
            idx += 1
            if idx == batch_size:
                if is_training:
                    yield (images, None)
                else:
                    yield images
                filenames = []
                images = np.zeros(batch_shape)
                idx = 0
        if idx > 0:
            if istraining:
                yield (images, None)
            else:
                yield images


def load_imagefiles(input_dir, batch_shape):
    images = np.zeros(batch_shape)
    filenames = []
    idx = 0
    batch_size = batch_shape[0]
    for filepath in tf.gfile.Glob(os.path.join(input_dir, '*.png')):
        with tf.gfile.Open(filepath, "rb") as f:
            image = imread(f, mode='RGB').astype(np.float) / 255.0
        # Images for inception classifier are normalized to be in [-1, 1] interval.
        images[idx, :, :, :] = image * 2.0 - 1.0
        filenames.append(os.path.basename(filepath))
        idx += 1
        if idx == batch_size:
            yield filenames, images
            filenames = []
            images = np.zeros(batch_shape)
            idx = 0
    if idx > 0:
        yield filenames, images

def load_patches(input_dir, patch_size=(64, 64)):
    patch_n = patch_size[0]
    while True:
        for filepath in tf.gfile.Glob(os.path.join(input_dir, '*.png')):
            with tf.gfile.Open(filepath, "rb") as f:
                image = imageio.imread(f).astype(np.float) / 255.0
              
            patches = extract_patches_2d(image, patch_size, max_patches=1024,random_state=2019)            
            
            for i in range(1024//patch_n):
                yield (patches[i*patch_n:(i+1)*patch_n,:], None)


def l2difference(x, x_adv):
    x = x.reshape(len(x), -1)
    x_adv = x_adv.reshape(len(x_adv), -1)
    num = np.linalg.norm(x - x_adv, axis=1)
    denom = np.linalg.norm(x, axis=1)
    return np.sum(num / denom) / len(x)

def sharpen(images):
    for i in range(len(images)):
        im = Image.fromarray(np.uint8(images[i,:]*255.),'RGB')
        im = im.filter(ImageFilter.SHARPEN)
        images[i,:] = np.asarray(im) / 255.

    return images

def smoothen(images):
    for i in range(len(images)):
        im = Image.fromarray(np.uint8(images[i,:]*255.),'RGB')
        im = im.filter(ImageFilter.SMOOTH_MORE)
        images[i,:] = np.asarray(im) / 255.

    return images
