import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.applications.vgg19 import VGG19


def read_image(file, w, h):
    image = cv2.imread(file)
    image = cv2.resize(image, (h, w))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


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


def get_model(model, name):
    def f(image):
        for layer in model.net.layers:
            image = layer(image)
            if layer.name == name:
                return image
    return f


nm_param = 1e-9
tv_param = 1e-9
learning_rate = 250000
epochs = 5000
epoch0 = 10
epoch1 = 1000
MEAN = np.float32([103.939, 116.779, 123.68])
STD = np.float32([1, 1, 1])

model = VGG19(include_top=True, weights="imagenet")
print([layer.name for layer in model.layers])
# model0 = get_model(model, "block5_conv1")
model0 = Model(inputs=model.input, outputs=model.get_layer("block5_conv1").output)

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
    X.assign_sub(dX * learning_rate)
    X.assign(clip(X))

    if _ % epoch0 == 0:
        print(_, loss.numpy())

    if _ % epoch1 == 0:
        plt.imshow(deprocess(X[0]))
        plt.axis("off")
        plt.savefig("image/%s.jpg" % _)

