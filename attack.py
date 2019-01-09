import os
from cleverhans.attacks import FastGradientMethod, BasicIterativeMethod
import numpy as np
import pandas as pd
from PIL import Image
from scipy.misc import imread
import imageio
import tensorflow as tf
from tensorflow.contrib.slim.nets import inception
from image_utils import load_images, load_imagefiles
from model_utils import InceptionModel
from io import BytesIO

FGSM = True
IFGSM = False  # Set one attack to True at a time
slim = tf.contrib.slim

checkpoint_path = "./inception-v3/inception_v3.ckpt"  # path to model checkpoint
input_dir = "./images/"
image_width = 299
image_height = 299
batch_size = 16

batch_shape = [batch_size, image_height, image_width, 3]
num_classes = 1001

epses = [0.005, 0.0135, 0.022, 0.0305, 0.039, 0.0475, 0.056, 0.0645,
         0.073, 0.0815, 0.09]

for epsilon in epses:
    with tf.Graph().as_default():

        x_input = tf.placeholder(tf.float32, shape=batch_shape)
        model = InceptionModel(num_classes)

        if FGSM:
            fgsm = FastGradientMethod(model)
            x_adv = fgsm.generate(x_input, eps=epsilon, clip_min=-1., clip_max=1.)
            try:
                output_dir = './fgsm_images_{}'.format(epsilon)
                os.mkdir(output_dir)

            except:
                pass

        if IFGSM:
            ifgsm = BasicIterativeMethod(model)
            ifgsm_params = {'eps': epsilon,
                            'eps_iter': epsilon / 5,
                            'nb_iter': 5,
                            'clip_min': -1.,
                            'clip_max': 1.,
                            }

            x_adv = ifgsm.generate(x_input, **ifgsm_params)
            try:
                output_dir = './ifgsm_images_{}'.format(epsilon)
                os.mkdir(output_dir)
            except:
                pass
        saver = tf.train.Saver(tf.contrib.slim.get_model_variables())
        session_creator = tf.train.ChiefSessionCreator(
            scaffold=tf.train.Scaffold(saver=saver),
            checkpoint_filename_with_path=checkpoint_path,
            master="")
        print('Attacking with epsilon {}'.format(epsilon))
        with tf.train.MonitoredSession(session_creator=session_creator) as sess:
            for filenames, images in load_imagefiles('./images', batch_shape):
                nontargeted_images = sess.run(x_adv, feed_dict={x_input: images})
                for filename, nontargeted_image in zip(filenames, nontargeted_images):
                    print("Writing {}".format(filename))
                    imageio.imwrite(os.path.join(output_dir, filename),
                                    np.uint8((nontargeted_image + 1.0) / 2.0 * 255.))
