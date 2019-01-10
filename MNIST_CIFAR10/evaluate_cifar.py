import scipy.io as sio
import numpy as np
import keras
from keras import backend
from models.mnistmodel import mnist_model
from models.cifarmodel import cifar_model
from loaddata import load_mnist, load_cifar
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
figure_title = "FGSM on CIFAR-10"
figure_filename = "FGSM_cifar.eps"

# CIFAR
# Load dataset
train_x, train_y, train_l = load_cifar()
test_x, test_y, test_l = load_cifar("test")
train_x = train_x.reshape([-1, 32, 32, 3])
test_x = test_x.reshape([-1, 32, 32, 3])

# Create TF session and set as Keras backend session
sess = tf.Session()
keras.backend.set_session(sess)

x = tf.placeholder(tf.float32, shape=(None, 32, 32, 3))
y = tf.placeholder(tf.float32, shape=(None, 10))

# cifar model
cifar_model = cifar_model()
optimizer = keras.optimizers.SGD(lr=1e-3, momentum=0.9, nesterov=False)
cifar_model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
cifar_model.load_weights("trained_model/cifar_model.h5")

# Load VAE
VAE_model = vae_model_cifar()
VAE_model.compile(optimizer='adam')
VAE_model.load_weights("vae_cifar.h5")


# Normalized L2 difference
def l2difference(x, x_adv):
    x = x.reshape(len(x), -1)
    x_adv = x_adv.reshape(len(x_adv), -1)
    num = np.linalg.norm(x - x_adv, axis=1)
    denom = np.linalg.norm(x, axis=1)
    return np.sum(num / denom) / len(x)


data_for_plotting = np.zeros([15, 7])
wrap = KerasModelWrapper(cifar_model)

# FGSM Attack & I-FGSM Attack

epsilons = np.linspace(0.005, 0.1, 15)
fgsm = FastGradientMethod(wrap, back='tf', sess=sess)
ifgsm = BasicIterativeMethod(wrap, back='tf', sess=sess)

for i in range(len(epsilons)):

    # FGSM
    fgsm_params = {'eps': epsilons[i],
                   'clip_min': 0.,
                   'clip_max': 1.,
                   }

    X_adv = np.zeros(test_x.shape)
    for j in range(10):
        k = j * 1000
        X_adv[k:k + 1000, :] = fgsm.generate_np(test_x[k:k + 1000, :], **fgsm_params)

    # Uncomment this for I-FGSM attack
    ''' 
    # I-FGSM
    ifgsm_params = {'eps': epsilons[i],
                    'eps_iter': epsilons[i] / 10,
                    'nb_iter': 10,
                    'clip_min': 0.,
                    'clip_max': 1.,
                    }
    X_adv = np.zeros(test_x.shape)
    for j in range(10):
        k = j * 1000
        X_adv[k:k + 1000, :] = ifgsm.generate_np(test_x[k:k + 1000, :], **ifgsm_params)
    '''

    # VAE
    X_vae = VAE_model.predict(X_adv, batch_size=500)

    # Jpeg
    X_jpeg = jpeg(X_adv, image_size=32, channels=3, quality=23)
    X_jpeg_10 = jpeg(X_adv, image_size=32, channels=3, quality=10)
    X_jpeg_50 = jpeg(X_adv, image_size=32, channels=3, quality=50)
    X_jpeg_70 = jpeg(X_adv, image_size=32, channels=3, quality=70)

    _, acc = cifar_model.evaluate(X_adv, test_y, batch_size=5000)
    _, acc_vae = cifar_model.evaluate(X_vae, test_y, batch_size=5000)
    _, acc_jpeg = cifar_model.evaluate(X_jpeg, test_y, batch_size=5000)

    _, acc_jpeg_10 = cifar_model.evaluate(X_jpeg_10, test_y, batch_size=5000, verbose=0)
    _, acc_jpeg_50 = cifar_model.evaluate(X_jpeg_50, test_y, batch_size=5000, verbose=0)
    _, acc_jpeg_70 = cifar_model.evaluate(X_jpeg_70, test_y, batch_size=5000, verbose=0)

    difference = l2difference(test_x, X_adv)
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
print(data_for_plotting)
plt.savefig(figure_filename, format='eps', dpi=1000)
