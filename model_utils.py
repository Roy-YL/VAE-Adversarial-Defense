
import tensorflow as tf
import numpy as np
from tensorflow.contrib.slim.nets import inception
from image_utils import jpeg
slim = tf.contrib.slim

class InceptionModel(object):
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.built = False

    def __call__(self, x_input):
        """Constructs model and return probabilities for given input."""
        reuse = True if self.built else None
        with slim.arg_scope(inception.inception_v3_arg_scope()):
            _, end_points = inception.inception_v3(
                x_input, num_classes=self.num_classes, is_training=False,
                reuse=reuse)
        self.built = True
        output = end_points['Predictions']
        probs = output.op.inputs[0]
        return probs


def print_accuracy(images, true_labels, jpeg_reconstruct=False, jpeg_quality=23):
    tf.reset_default_graph()
    with tf.Graph().as_default():  # Prepare graph
        x_input = tf.placeholder(tf.float32, shape=[100, 299, 299, 3])

        with slim.arg_scope(inception.inception_v3_arg_scope()):
            _, end_points = inception.inception_v3(x_input, num_classes=1001, is_training=False)

        predicted_labels = tf.argmax(end_points['Predictions'], 1)

        # Run computation
        saver = tf.train.Saver(slim.get_model_variables())
        session_creator = tf.train.ChiefSessionCreator(
            scaffold=tf.train.Scaffold(saver=saver),
            checkpoint_filename_with_path='./inception-v3/inception_v3.ckpt')

        with tf.train.MonitoredSession(session_creator=session_creator) as sess:
            if jpeg_reconstruct:
                images = (images + 1.0) / 2.0
                images = jpeg(images, quality=jpeg_quality)
                images = images * 2.0 - 1.0
            labels = np.zeros(1000)
            for i in range(10):
                labels[i * 100:(i + 1) * 100] = sess.run(predicted_labels,
                                                         feed_dict={x_input: images[i * 100:(i + 1) * 100, :]})

    accuracy = np.sum(labels == true_labels) / len(labels)
    return accuracy


def reconstruct_image(images, model, patch_size=32, step_size=10):
    moves = int((299 - (patch_size - step_size)) / step_size) + 1
    output_image = np.zeros(images.shape)
    repeats = np.zeros(images.shape)

    for i in range(moves):
        for j in range(moves):
            if i == moves - 1 and j == moves - 1:
                output_image[:, 299 - patch_size:299, 299 - patch_size:299, :] += model.predict(
                    images[:, 299 - patch_size:299, 299 - patch_size:299, :])
                repeats[:, 299 - patch_size:299, 299 - patch_size:299, :] += 1
                continue
            if i == (moves - 1):
                output_image[:, 299 - patch_size:299, j * step_size:j * step_size + patch_size, :] += model.predict(
                    images[:, 299 - patch_size:299, j * step_size:j * step_size + patch_size, :])
                repeats[:, 299 - patch_size:299, j * step_size:j * step_size + patch_size, :] += 1
                continue
            if j == (moves - 1):
                output_image[:, i * step_size:i * step_size + patch_size, 299 - patch_size:299, :] += model.predict(
                    images[:, i * step_size:i * step_size + patch_size, 299 - patch_size:299, :])
                repeats[:, i * step_size:i * step_size + patch_size, 299 - patch_size:299, :] += 1
                continue

            output_image[:, i * step_size:i * step_size + patch_size, j * step_size:j * step_size + patch_size,
            :] += model.predict(
                images[:, i * step_size:i * step_size + patch_size, j * step_size:j * step_size + patch_size, :])
            repeats[:, i * step_size:i * step_size + patch_size, j * step_size:j * step_size + patch_size, :] += 1

    return output_image / repeats
