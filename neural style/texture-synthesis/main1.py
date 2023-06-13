import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.applications.vgg19 import VGG19

"""
TEXTURE_LAYERS = [
    "block1_conv1", "block1_conv2", "block1_pool", 
    "block2_conv1", "block2_conv2", "block2_pool", 
    "block3_conv1", "block3_conv2", "block3_conv3", "block3_conv4", "block3_pool", 
    "block4_conv1", "block4_conv2", "block4_conv3", "block4_conv4", "block4_pool", 
    "block5_conv1", "block5_conv2", "block5_conv3", "block5_conv4", "block5_pool",
]
TEXTURE_WEIGHTS = [
    1, 1, 1,
    1, 1, 1,
    1, 1, 1, 1, 1,
    1, 1, 1, 1, 1,
    1, 1, 1, 1, 1,
]
"""

MEAN = np.float32([103.939, 116.779, 123.68])
STD = np.float32([1, 1, 1])
TEXTURE_LAYERS = ["block1_conv1", "block2_conv1", "block3_conv1", "block4_conv1", "block5_conv1"]
TEXTURE_WEIGHTS = [1, 1, 1, 1, 1]
epochs = 1000
epoch0 = 10
epoch1 = 100
learning_rate = 3


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


model = VGG19(include_top=True, weights="imagenet")
model.trainable = False
model0 = Model(inputs=model.input, outputs=[model.get_layer(name).output for name in TEXTURE_LAYERS])
model0.trainable = False
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

X0 = read_image("input/0.jpg", 224, 224)
X0 = process(X0)
X0 = X0[None]
gram_matrixs0 = get_gram_matrixs(model0, X0)

X = 255 * np.random.rand(224, 224, 3)
X = process(X)
X = X[None]
X = tf.Variable(X)
for _ in range(epochs):
    with tf.GradientTape() as tape:
        tape.watch(X)
        gram_matrixs = get_gram_matrixs(model0, X)
        loss = get_texture_loss(gram_matrixs, gram_matrixs0, TEXTURE_WEIGHTS)
    dX = tape.gradient(loss, X)
    optimizer.apply_gradients([(dX, X)])
    X.assign(clip(X))

    if _ % epoch0 == 0:
        print(_, loss.numpy())

    if _ % epoch1 == 0:
        plt.imshow(deprocess(X[0]))
        plt.axis("off")
        plt.savefig("image/%s.jpg" % _)

