import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import Model
from cs231n.classifiers.squeezenet import SqueezeNet
from cs231n.image_utils import preprocess_image, deprocess_image
from cs231n.image_utils import SQUEEZENET_MEAN, SQUEEZENET_STD

TEXTURE_LAYERS = [0, 3, 5, 6]
TEXTURE_WEIGHTS = [300000, 1000, 15, 3]
epochs = 200
epoch0 = 10
epoch1 = 20
learning_rate = 3


def read_image(file, w, h):
    image = cv2.imread(file)
    image = cv2.resize(image, (h, w))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def get_gram_matrix(output):
    shape = tf.shape(output)
    F = tf.reshape(output, (shape[0], shape[1] * shape[2], shape[3]))
    N = tf.cast(2 * shape[1] * shape[2], tf.float32)
    G = tf.matmul(F, F, transpose_a=True) / N
    return G


def get_gram_matrixs(model, X):
    outputs = model(X)
    return [get_gram_matrix(output) for output in outputs]


def get_texture_loss(gram_matrixs1, gram_matrixs2, texture_weights):
    loss = 0
    for w, m1, m2 in zip(texture_weights, gram_matrixs1, gram_matrixs2):
        loss += w * tf.nn.l2_loss(m1 - m2)
    return loss


SAVE_PATH = "cs231n/datasets/squeezenet.ckpt"
if not os.path.exists(SAVE_PATH + ".index"):
    raise ValueError("You need to download SqueezeNet!")
model = SqueezeNet()
model.load_weights(SAVE_PATH)
model.trainable = False
model0 = Model(
    inputs=model.net.input,
    outputs=[layer.output for index, layer in enumerate(model.net.layers) if index in TEXTURE_LAYERS]
)
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

X0 = read_image("input/0.jpg", 224, 224)
X0 = preprocess_image(X0)
X0 = X0[None]
gram_matrixs0 = get_gram_matrixs(model0, X0)

X = 255 * np.random.rand(224, 224, 3)
X = preprocess_image(X)
X = X[None]
X = tf.Variable(X)
for _ in range(epochs):
    with tf.GradientTape() as tape:
        tape.watch(X)
        gram_matrixs = get_gram_matrixs(model0, X)
        loss = get_texture_loss(gram_matrixs, gram_matrixs0, TEXTURE_WEIGHTS)
    dX = tape.gradient(loss, X)
    optimizer.apply_gradients([(dX, X)])
    X.assign(tf.clip_by_value(X, -SQUEEZENET_MEAN / SQUEEZENET_STD, (1.0 - SQUEEZENET_MEAN) / SQUEEZENET_STD))

    if _ % epoch0 == 0:
        print(_, loss.numpy())

    if _ % epoch1 == 0:
        plt.imshow(deprocess_image(X[0]))
        plt.axis("off")
        plt.savefig("image/%s.jpg" % _)

