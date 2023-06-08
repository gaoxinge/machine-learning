import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.applications.vgg19 import VGG19


def process(image):
    """based on keras.applications.vgg19.preprocess_input"""
    image = image[..., ::-1]
    return (image.astype(np.float32) - MEAN) / STD


def deprocess(image):
    """based on keras.applications.vgg19.preprocess_input"""
    image = image * STD + MEAN
    image = image[..., ::-1]
    return np.clip(image, 0, 255).astype(np.uint8)


def clip(image):
    return tf.clip_by_value(image, clip_value_min=0, clip_value_max=255)


l2_reg = 1e-8
learning_rate = 250000
epochs = 1000
epoch0 = 10
epoch1 = 200
target = 76
MEAN = np.float32([103.939, 116.779, 123.68])
STD = np.float32([1, 1, 1])

model = VGG19(include_top=True, weights="imagenet")
model.trainable = False

X = 255 * np.random.rand(224, 224, 3)
X = process(X)[None]
X = tf.Variable(X)
for _ in range(epochs):
    with tf.GradientTape() as tape:
        tape.watch(X)
        loss = model(X)[0, target] - l2_reg * tf.nn.l2_loss(X)
    dX = tape.gradient(loss, X)
    X.assign_add(dX * learning_rate)
    X.assign(clip(X))

    if _ % epoch0 == 0:
        print(_, loss.numpy())

    if _ % epoch1 == 0:
        plt.imshow(deprocess(X[0]))
        plt.axis("off")
        plt.savefig("image/%s.jpg" % _)

