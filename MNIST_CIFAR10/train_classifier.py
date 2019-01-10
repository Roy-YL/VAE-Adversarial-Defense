from loaddata import load_cifar, load_mnist
from models.mnistmodel import mnist_model
from models.cifarmodel import cifar_model
import tensorflow as tf
from sklearn.metrics import confusion_matrix
from keras.callbacks import LearningRateScheduler
import numpy as np
import keras

# MNIST
X_train, Y_train, X_test, Y_test, labels_train, labels_test = load_mnist()

# CIFAR
# Load dataset
train_x, train_y, train_l = load_cifar()
test_x, test_y, test_l = load_cifar("test")

# Reshape
train_x = train_x.reshape([-1, 32, 32, 3])
test_x = test_x.reshape([-1, 32, 32, 3])

# Create TF session and set as Keras backend session
sess = tf.Session()
keras.backend.set_session(sess)

# Construct model
mnist_classifier = mnist_model()
cifar_classifier = cifar_model()


# Set parameters
def lr_schedule(epoch):
    lrate = 0.001
    if epoch > 75:
        lrate = 0.0005
    elif epoch > 100:
        lrate = 0.0003
    return lrate


optimizer = keras.optimizers.SGD(lr=1e-3, momentum=0.9, nesterov=False)
cifar_classifier.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
mnist_classifier.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# Train MNIST classifier
try:
    mnist_classifier.load_weights("trained_model/mnist_model.h5")
    print("Successfully loaded weights")
except:
    print("Fail to load weights, start training..")
    history = mnist_classifier.fit(X_train, Y_train, batch_size=256, epochs=150, validation_data=(X_test, Y_test),
                                   callbacks=[LearningRateScheduler(lr_schedule)], verbose=2)
    mnist_classifier.save_weights("trained_model/mnist_model.h5")

# Train CIFAR classifier
try:
    cifar_classifier.load_weights("trained_model/cifar_model.h5")
    print("Successfully loaded weights")
except:
    print("Fail to load weights, start training..")
    history = cifar_classifier.fit(train_x, train_y, batch_size=256, epochs=150, validation_data=(test_x, test_y),
                                   callbacks=[LearningRateScheduler(lr_schedule)], verbose=2)
    cifar_classifier.save_weights("trained_model/cifar_model.h5")
