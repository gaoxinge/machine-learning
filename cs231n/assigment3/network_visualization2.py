import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.backend import get_session
from scipy.ndimage.filters import gaussian_filter1d
from cs231n.classifiers.squeezenet import SqueezeNet
from cs231n.image_utils import preprocess_image, deprocess_image
from cs231n.image_utils import SQUEEZENET_MEAN, SQUEEZENET_STD
from cs231n.data_utils import load_imagenet_val
X_raw, y, class_names = load_imagenet_val(num=5)


def blur_image(X, sigma=1):
    X = gaussian_filter1d(X, sigma, axis=1)
    X = gaussian_filter1d(X, sigma, axis=2)
    return X


def jitter(X, ox, oy):
    return np.roll(np.roll(X, ox, 1), oy, 2)


def create_class_visualization(target_y, model, **kwargs):
    """
    Generate an image to maximize the score of target_y under a pretrained model.
    
    Inputs:
    - target_y: Integer in the range [0, 1000) giving the index of the class
    - model: A pretrained CNN that will be used to generate the image
    
    Keyword arguments:
    - l2_reg: Strength of L2 regularization on the image
    - learning_rate: How big of a step to take
    - num_iterations: How many iterations to use
    - blur_every: How often to blur the image as an implicit regularizer
    - max_jitter: How much to jitter the image as an implicit regularizer
    - show_every: How often to show the intermediate result
    """
    l2_reg = kwargs.pop('l2_reg', 1e-3)
    learning_rate = kwargs.pop('learning_rate', 25)
    num_iterations = kwargs.pop('num_iterations', 100)
    blur_every = kwargs.pop('blur_every', 10)
    max_jitter = kwargs.pop('max_jitter', 16)
    show_every = kwargs.pop('show_every', 25)

    X = 255 * np.random.rand(224, 224, 3)
    X = preprocess_image(X)[None]

    sess = get_session()
    for t in range(num_iterations):
        ox, oy = np.random.randint(0, max_jitter, 2)
        X = jitter(X, ox, oy)

        Y = tf.convert_to_tensor(X)
        with tf.GradientTape() as tape:
            #Y = tf.convert_to_tensor(X)
            tape.watch(Y)
            loss = model(Y)[0, target_y] - l2_reg * tf.nn.l2_loss(Y)
            #dY = tape.gradient(loss, Y)
            #dX = sess.run(dY)
            #X += dX[0] * learning_rate
        dY = tape.gradient(loss, Y)
        dX = sess.run(dY)
        X += dX[0] * learning_rate

        X = jitter(X, -ox, -oy)
        X = np.clip(X, -SQUEEZENET_MEAN/SQUEEZENET_STD, (1.0 - SQUEEZENET_MEAN)/SQUEEZENET_STD)
        if t % blur_every == 0:
            X = blur_image(X, sigma=0.5)

        if t == 0 or (t + 1) % show_every == 0 or t == num_iterations - 1:
            plt.imshow(deprocess_image(X[0]))
            class_name = class_names[target_y]
            plt.title('%s\nIteration %d / %d' % (class_name, t + 1, num_iterations))
            plt.gcf().set_size_inches(4, 4)
            plt.axis('off')
            plt.savefig("%s.jpg" % t)
    return X


SAVE_PATH = "cs231n/datasets/squeezenet.ckpt"
if not os.path.exists(SAVE_PATH + ".index"):
    raise ValueError("You need to download SqueezeNet!")
model = SqueezeNet()
model.load_weights(SAVE_PATH)
model.trainable = False

target_y = 76 # Tarantula
out = create_class_visualization(target_y, model)                    
