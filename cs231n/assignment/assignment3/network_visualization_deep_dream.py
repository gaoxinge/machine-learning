import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from cs231n.classifiers.squeezenet import SqueezeNet
from cs231n.image_utils import preprocess_image, deprocess_image
from cs231n.image_utils import SQUEEZENET_MEAN, SQUEEZENET_STD
from cs231n.net_visualization_tensorflow import jitter, blur_image


def read_image(file, w, h):
    image = cv2.imread(file)
    image = cv2.resize(image, (h, w))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


learning_rate = 0.01
epochs = 1000
epoch0 = 10
epoch1 = 200

SAVE_PATH = "cs231n/datasets/squeezenet.ckpt"
if not os.path.exists(SAVE_PATH + ".index"):
    raise ValueError("You need to download SqueezeNet!")
model = SqueezeNet()
model.load_weights(SAVE_PATH)
model.trainable = False

X = read_image("1.png", 224, 224)
X = preprocess_image(X)[None]
X = tf.Variable(X)
for _ in range(epochs):
    ox, oy = np.random.randint(0, 16, 2)
    X = jitter(X, ox, oy)

    with tf.GradientTape() as tape:
        tape.watch(X)
        Y = model(X)
        loss = tf.math.reduce_sum(Y)
    dX = tape.gradient(loss, X)
    dX /= tf.math.reduce_std(dX) + 1e-8
    X += learning_rate * dX

    X = jitter(X, -ox, -oy)
    X = tf.clip_by_value(X, -SQUEEZENET_MEAN / SQUEEZENET_STD, (1.0 - SQUEEZENET_MEAN) / SQUEEZENET_STD)
    if _ % 10 == 0:
        X = blur_image(X, sigma=0.5)

    if _ % epoch0 == 0:
        print(_, loss.numpy())

    if _ % epoch1 == 0:
        plt.imshow(deprocess_image(X[0]))
        plt.axis("off")
        plt.savefig("image/%s.jpg" % _)

