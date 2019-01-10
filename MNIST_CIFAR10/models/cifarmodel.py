import keras
from keras import backend
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Conv2D, BatchNormalization, MaxPooling2D, Dropout
from keras import regularizers



def cifar_model(logits=False, input_ph=None, img_rows=32, img_cols=32,
              channels=3, nb_filters=64, nb_classes=10):
    """
    Defines a CNN model using Keras sequential model
    :param logits: If set to False, returns a Keras model, otherwise will also
                    return logits tensor
    :param input_ph: The TensorFlow tensor for the input
                    (needed if returning logits)
                    ("ph" stands for placeholder but it need not actually be a
                    placeholder)
    :param img_rows: number of row in the image
    :param img_cols: number of columns in the image
    :param channels: number of color channels (e.g., 1 for MNIST)
    :param nb_filters: number of convolutional filters per layer
    :param nb_classes: the number of output classes
    :return:
    """
    model = Sequential()

    # Define the layers successively (convolution layers are version dependent)
    if keras.backend.image_dim_ordering() == 'th':
        input_shape = (channels, img_rows, img_cols)
    else:
        input_shape = (img_rows, img_cols, channels)

    layers = [Conv2D(32, (3, 3), padding="same",input_shape=input_shape,kernel_regularizer=regularizers.l2(1e-4)),
              BatchNormalization(axis = -1),
              Activation('relu'),
              Conv2D(32, (3, 3), padding="same",input_shape=input_shape,kernel_regularizer=regularizers.l2(1e-4)),
              BatchNormalization(axis = -1),
              Activation('relu'),
              MaxPooling2D(pool_size=(2,2)),
              Dropout(.2),
              
              Conv2D(64, (3, 3), padding="same",input_shape=input_shape,kernel_regularizer=regularizers.l2(1e-4)),
              BatchNormalization(axis = -1),
              Activation('relu'),
              Conv2D(64, (3, 3), padding="same",input_shape=input_shape,kernel_regularizer=regularizers.l2(1e-4)),
              BatchNormalization(axis = -1),
              Activation('relu'),
              MaxPooling2D(pool_size=(2,2)),
              Dropout(.3),
              
              Conv2D(128, (3, 3), padding="same",input_shape=input_shape,kernel_regularizer=regularizers.l2(1e-4)),
              BatchNormalization(axis = -1),
              Activation('relu'),
              Conv2D(128, (3, 3), padding="same",input_shape=input_shape,kernel_regularizer=regularizers.l2(1e-4)),
              BatchNormalization(axis = -1),
              Activation('relu'),
              MaxPooling2D(pool_size=(2,2)),
              Dropout(.4),
              
              Flatten(),
              Dense(nb_classes)
              ]

    for layer in layers:
        model.add(layer)

    if logits:
        logits_tensor = model(input_ph)
    model.add(Activation('softmax'))

    if logits:
        return model, logits_tensor
    else:
        return model

#cifar_model().summary()
