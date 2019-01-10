import scipy.io as sio
import numpy as np
import keras
from keras import backend
from models.mnistmodel import mnist_model
from loaddata import load_mnist
import tensorflow as tf
import matplotlib.pyplot as plt
from models.vae import vae_model_mnist, vae_model_cifar
from jpeg import jpeg
from cleverhans.attacks import FastGradientMethod
from cleverhans.utils_keras import KerasModelWrapper
from cleverhans.attacks import BasicIterativeMethod

plt.switch_backend('agg')

FGSM = True
IFGSM = False  # Set one attack to True at a time
figure_title = "FGSM on MNIST"
figure_filename = "FGSM_mnist.eps"

# MNIST
X_train, Y_train, X_test, Y_test, labels_train, labels_test = load_mnist()

# Create TF session and set as Keras backend session
sess = tf.Session()
keras.backend.set_session(sess)

x = tf.placeholder(tf.float32, shape=(None, 28, 28, 1))
y = tf.placeholder(tf.float32, shape=(None, 10))

# Mnist model
mnist_model, logits = mnist_model(input_ph=x, logits=True)
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
mnist_model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
mnist_model.load_weights("trained_model/mnist_model.h5")

# Load VAE
VAE_model = vae_model_mnist()
VAE_model.compile(optimizer='adam')
VAE_model.load_weights("vae_mnist.h5")


# Normalized L2 difference
def l2difference(x, x_adv):
    x = x.reshape(len(x), -1)
    x_adv = x_adv.reshape(len(x_adv), -1)
    num = np.linalg.norm(x - x_adv, axis=1)
    denom = np.linalg.norm(x, axis=1)
    return np.sum(num / denom) / len(x)


data_for_plotting = np.zeros([20, 7])
wrap = KerasModelWrapper(mnist_model)

# FGSM Attack
epsilons = np.linspace(0.005, 0.12, 20)
fgsm = FastGradientMethod(wrap, back='tf', sess=sess)
ifgsm = BasicIterativeMethod(wrap, back='tf', sess=sess)

for i in range(len(epsilons)):

    # FGSM
    fgsm_params = {'eps': epsilons[i],
                   'clip_min': 0.,
                   'clip_max': 1.,
                   }
    adv_x = fgsm.generate(x, **fgsm_params)

    X_adv = np.zeros(X_test.shape)
    for j in range(10):
        k = j * 1000
        X_adv[k:k + 1000, :] = fgsm.generate_np(X_test[k:k + 1000, :], **fgsm_params)

    # Uncomment this for I-FGSM attack
    '''  
    # I-FGSM
    ifgsm_params = {'eps':epsilons[i],
                    'eps_iter':epsilons[i]/10,
                    'nb_iter':10,
                    'clip_min':0.,
                    'clip_max':1.,
    }
    X_adv = np.zeros(X_test.shape)
    for j in range(10):
        k = j*1000
        X_adv[k:k+1000,:] = ifgsm.generate_np(X_test[k:k+1000,:], **ifgsm_params)
    '''

    # VAE
    X_vae = VAE_model.predict(X_adv)

    # Jpeg
    X_jpeg = jpeg(X_adv, image_size=28, channels=1, quality=23)
    X_jpeg_10 = jpeg(X_adv, image_size=28, channels=1, quality=10)
    X_jpeg_50 = jpeg(X_adv, image_size=28, channels=1, quality=50)
    X_jpeg_70 = jpeg(X_adv, image_size=28, channels=1, quality=70)

    _, acc = mnist_model.evaluate(X_adv, Y_test, batch_size=5000, verbose=0)
    _, acc_vae = mnist_model.evaluate(X_vae, Y_test, batch_size=5000, verbose=0)
    _, acc_jpeg = mnist_model.evaluate(X_jpeg, Y_test, batch_size=5000, verbose=0)
    _, acc_jpeg_10 = mnist_model.evaluate(X_jpeg_10, Y_test, batch_size=5000, verbose=0)
    _, acc_jpeg_50 = mnist_model.evaluate(X_jpeg_50, Y_test, batch_size=5000, verbose=0)
    _, acc_jpeg_70 = mnist_model.evaluate(X_jpeg_70, Y_test, batch_size=5000, verbose=0)

    difference = l2difference(X_test, X_adv)
    data_for_plotting[i, :] = [difference, acc, acc_vae, acc_jpeg_10, acc_jpeg, acc_jpeg_50, acc_jpeg_70]

fig, ax1 = plt.subplots()

ax1.plot(data_for_plotting[:, 0], data_for_plotting[:, 1], "b-", label="No Defense")
ax1.plot(data_for_plotting[:, 0], data_for_plotting[:, 2], "r-", label="VAE")
ax1.plot(data_for_plotting[:, 0], data_for_plotting[:, 3], "y-", label="JPEG-10")
ax1.plot(data_for_plotting[:, 0], data_for_plotting[:, 4], "g-", label="JPEG-23")
ax1.plot(data_for_plotting[:, 0], data_for_plotting[:, 5], "m-", label="JPEG-50")
ax1.plot(data_for_plotting[:, 0], data_for_plotting[:, 6], "k-", label="JPEG-70")

ax1.set_xlabel("Normalized l2 Difference")
ax1.set_ylabel("Top-1 Accuracy", color='r')

ax1.legend()
plt.title(figure_title)
plt.savefig(figure_filename, format='eps', dpi=1000)
