import os
import numpy as np
from scipy.misc import imread, imresize
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.applications import VGG16
from cs231n.classifiers.squeezenet import SqueezeNet
tf.enable_eager_execution()


def read_image(file, w, h):
    image = imread(file)
    image = imresize(image, (w, h)) 
    return image


def process(image):
    return (image.astype(np.float32) / 255 - MEAN) / STD


def deprocess(image):
    image = 255 * (image * STD + MEAN)
    return np.clip(image, 0, 255).astype(np.uint8)


def f(model, name):
    def g(image):
        for layer in model.net.layers:
            image = layer(image)
            if layer.name == name:
               return image
    return g


nm_param = 1e-5
tv_param = 1e-5
learning_rate = 25
epochs = 5000
epoch0 = 100
epoch1 = 1000
MEAN = np.float32([0.485, 0.456, 0.406])
STD = np.float32([0.229, 0.224, 0.225])

model = VGG16(include_top=False, weights="imagenet")
print([layer.name for layer in model.layers])
model0 = Model(inputs=model.input, outputs=model.get_layer("block5_conv1").output)

"""
SAVE_PATH = "cs231n/datasets/squeezenet.ckpt"
if not os.path.exists(SAVE_PATH + ".index"):
    raise ValueError("You need to download SqueezeNet!")
model = SqueezeNet()
model.load_weights(SAVE_PATH)
model.trainable = False
print([layer.name for layer in model.net.layers])
model0 = f(model, "classifier/layer1")
model0 = Model(inputs=mode.net.input, output=model.net.get_layer("classifier/layer1").output)
"""

X0 = read_image("monkey.jpg", 224, 224)
X0 = process(X0)
X0 = X0[None]
phi0 = model0(X0)
phi1 = tf.nn.l2_loss(phi0)


X = 255 * np.random.rand(224, 224, 3)
X = process(X)
X = X[None]
X = tf.Variable(X)
for _ in range(epochs):
    with tf.GradientTape() as tape:
         tape.watch(X)
         loss = tf.nn.l2_loss(model0(X) - phi0) / phi1
         loss += nm_param * tf.nn.l2_loss(X)
         loss += tv_param * tf.reduce_sum(tf.image.total_variation(X))
    dX = tape.gradient(loss, X)
    X.assign_sub(dX[0] * learning_rate)

    if _ % epoch0 == 0:
        print(_, loss.numpy())

    if _ % epoch1 == 0:
        plt.imshow(deprocess(X[0]))
        plt.axis("off")
        plt.savefig("image/%s.jpg" % _)
