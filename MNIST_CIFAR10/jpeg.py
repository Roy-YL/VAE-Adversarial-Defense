from PIL import Image
from io import BytesIO
import numpy as np
from loaddata import load_mnist, load_cifar


def jpeg(X, image_size=32, channels=3, quality=75):
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
