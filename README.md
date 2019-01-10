# VAE-Adversarial-Defense

Code to produce the results of ArXiv preprint ["Adversarial Defense of Image Classification Using a Variational Auto-Encoder"](https://arxiv.org/abs/1812.02891).

### Requirements

- Python 3.6
- Tensorflow and Keras
- Cleverhans
- Sklearn
- Scipy, Imageio, matplotlib

### MNIST and CIFAR-10

The MNIST dataset should be downloaded by the user and stored under `data` directory in `mat` format. [MNIST](http://yann.lecun.com/exdb/mnist/)

The CIFAR-10 dataset can be downloaded by running provided script. [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html)

To train the classifiers, run 

```shell
python train_classifier.py
```

To train the VAEs, run

```shell
python train_vae.py
```

To evaluate the attacks and defenses, run

```shell
python evaluate_mnist.py
```

and

```shell
python evaluate_cifar.py
```

### NIPS 2017 Defense Against Adversarial Attacks Dataset

Download the 1000 image dataset and pretrained Inception-V3 model checkpoint from the [Kaggle competition](https://www.kaggle.com/c/nips-2017-defense-against-adversarial-attack/data).

Store the images in a directory named `images`, the Inception-V3 model checkpoint in a directory named `inception-v3`.

To train the VAE models on the images, run

```shell
python train_vae.py
```

To perform FGSM and I_FGSM attacks on the images, run

```shell
python attack.py
```

The attacked images will be stored in directories with names such as `fgsm_images_0.005` where `0.005` indicates the attack hyperparameter `epsilon`.

To evaluate the defense on the attacked images, run

```shell
python evaluate.py
```

The results will be saved into a `csv` file.