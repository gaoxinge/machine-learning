import os
import numpy as np
from scipy.misc import imread, imresize
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import Model
from cs231n.classifiers.squeezenet import SqueezeNet
tf.enable_eager_execution()


MEAN = np.float32([0.485, 0.456, 0.406])
STD = np.float32([0.229, 0.224, 0.225])
TEXTURE_LAYERS = [0, 3, 5, 6]
TEXTURE_WEIGHTS = [20000, 500, 12, 1]
epochs = 10000
epoch0 = 10
epoch1 = 100
learning_rate = 0.02
beta1 = 0.99
epsilon = 1e-1


def read_image(file, w, h):
    image = imread(file)
    image = imresize(image, (w, h))
    return image


def process(image):
    return (image.astype(np.float32) / 255 - MEAN) / STD


def deprocess(image):
    image = 255 * (image * STD + MEAN)
    return np.clip(image, 0, 255).astype(np.uint8)


def clip(image):
    return tf.clip_by_value(image, clip_value_min=0, clip_value_max=1) 


def get_gram_matrix(output):
    shape = tf.shape(output)
    F = tf.reshape(output, (shape[0], shape[1] * shape[2], shape[3]))
    N = tf.cast(2 * shape[1] * shape[2], tf.float32)
    G = tf.matmul(F, F, transpose_a=True) / N
    return G


def get_gram_matrixs(model, X):
    outputs = model(X)
    return [get_gram_matrix(output) for output in outputs]


def get_texture_loss(gram_matrixs1, gram_matrixs2, texture_weights=TEXTURE_WEIGHTS):
    loss = 0
    for w, m1, m2 in zip(texture_weights, gram_matrixs1, gram_matrixs2):
        loss +=  w * tf.nn.l2_loss(m1 - m2)
    return loss


SAVE_PATH = "cs231n/datasets/squeezenet.ckpt"
if not os.path.exists(SAVE_PATH + ".index"):
    raise ValueError("You need to download SqueezeNet!")
model = SqueezeNet()
model.load_weights(SAVE_PATH)
model.trainable = False
model0 = Model(inputs=model.net.input, outputs=[layer.output for index, layer in enumerate(model.net.layers) if index in TEXTURE_LAYERS])
model.trainable = False

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=beta1, epsilon=epsilon)

X0 = read_image("upload/1.jpg", 224, 224)
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
        loss = get_texture_loss(gram_matrixs, gram_matrixs0)
    dX = tape.gradient(loss, X)
    optimizer.apply_gradients([(dX, X)])
    # X.assign_sub(dX[0] * learning_rate)
    X.assign(clip(X))

    if _ % epoch0 == 0:
        print(_, loss.numpy())

    if _ % epoch1 == 0:
        plt.imshow(deprocess(X[0]))
        plt.axis("off")
        plt.savefig("image/%s.jpg" % _)
