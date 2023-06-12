import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import Model
from cs231n.classifiers.squeezenet import SqueezeNet
from cs231n.image_utils import preprocess_image, deprocess_image
from cs231n.image_utils import SQUEEZENET_MEAN, SQUEEZENET_STD
from cs231n.net_visualization_tensorflow import jitter, blur_image


def read_image(file, w, h):
    image = cv2.imread(file)
    image = cv2.resize(image, (h, w))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def get_model(model, name):
    def f(image):
        for layer in model.net.layers:
            image = layer(image)
            if layer.name == name:
                return image
    return f


nm_param = 1e-5
tv_param = 1e-5
learning_rate = 25
epochs = 5000
epoch0 = 10
epoch1 = 1000

SAVE_PATH = "cs231n/datasets/squeezenet.ckpt"
if not os.path.exists(SAVE_PATH + ".index"):
    raise ValueError("You need to download SqueezeNet!")
model = SqueezeNet()
model.load_weights(SAVE_PATH)
model.trainable = False
print([layer.name for layer in model.net.layers])
# model0 = get_model(model, "classifier/layer1")
model0 = Model(inputs=model.net.input, outputs=model.net.get_layer("classifier/layer1").output)

X0 = read_image("monkey.jpg", 224, 224)
X0 = preprocess_image(X0)
X0 = X0[None]
phi0 = model0(X0)
phi1 = tf.nn.l2_loss(phi0)

X = 255 * np.random.rand(224, 224, 3)
X = preprocess_image(X)
X = X[None]
X = tf.Variable(X)

for _ in range(epochs):
    ox, oy = np.random.randint(0, 16, 2)
    if ox == 0 and oy == 0:
        continue
    X = jitter(X, ox, oy)

    with tf.GradientTape() as tape:
        tape.watch(X)
        loss = tf.nn.l2_loss(model0(X) - phi0) / phi1
        loss += nm_param * tf.nn.l2_loss(X)
        loss += tv_param * tf.reduce_sum(tf.image.total_variation(X))
    dX = tape.gradient(loss, X)
    X -= dX * learning_rate

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

