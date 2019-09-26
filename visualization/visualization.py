import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.applications import VGG16
from cs231n.classifiers.squeezenet import SqueezeNet
tf.enable_eager_execution()


def process(image):
    return (image.astype(np.float32) / 255 - MEAN) / STD


def deprocess(image):
     image = 255 * (image * STD + MEAN)
     return np.clip(image, 0, 255).astype(np.uint8)


l2_reg = 1e-5
learning_rate = 25
epochs = 5000
epoch0 = 10
epoch1 = 1000
target = 76
MEAN = np.float32([0.485, 0.456, 0.406])
STD = np.float32([0.229, 0.224, 0.225])

"""
model = VGG16(include_top=True, weights="imagenet")
"""

SAVE_PATH = "cs231n/datasets/squeezenet.ckpt"
if not os.path.exists(SAVE_PATH + ".index"):
    raise ValueError("You need to download SqueezeNet!")
model = SqueezeNet()
model.load_weights(SAVE_PATH)
model.trainable = False

X = 255 * np.random.rand(224, 224, 3)
X = process(X)
X = X[None]
X = tf.Variable(X)
for _ in range(epochs):
    with tf.GradientTape() as tape:
        tape.watch(X)
        loss = model(X)[0, target] - l2_reg * tf.nn.l2_loss(X)
    dX = tape.gradient(loss, X)
    X.assign_add(dX[0] * learning_rate)
    
    if _ % epoch0 == 0:
        print(_, loss.numpy())

    if _ % epoch1 == 0:
        plt.imshow(deprocess(X[0]))
        plt.axis("off")
        plt.savefig("image/%s.jpg" % _)
