import numpy as np
import tensorflow as tf
from PIL import Image
tf.enable_eager_execution()


def make_kernel(a):
    a = tf.constant(a, dtype=tf.float32)
    a = tf.tile(a[:, :, tf.newaxis, tf.newaxis], [1, 1, 3, 3])
    return a


def simple_conv(x, k):
    x = tf.expand_dims(x, 0)
    y = tf.nn.conv2d(input=x, filter=k, strides=[1, 1, 1, 1], padding="SAME")
    return tf.squeeze(y)


def laplace(x):
    laplace_k = make_kernel([[0.5, 1.0, 0.5],
                             [1.0, -6.0, 1.0],
                             [0.5, 1.0, 0.5]])
    return simple_conv(x, laplace_k)


def save(a, name):
    a = np.clip(a, 0, 255).astype(np.uint8)
    image = Image.fromarray(a)
    image.save(name)


h = 0.03
steps = 100
step0 = 5

with Image.open("a.jpg") as image:
    U = np.array(image)
    U = U.astype(np.float32)
    U = tf.Variable(U)
    d1, d2 = image.size
    Ut = np.zeros([d2, d1, 3], dtype=np.float32)

for _ in range(steps):
    U = U + Ut * h
    Ut = Ut + laplace(U) * h
    if _ % step0 == 0:
       save(U.numpy(), "image/%s.jpg" % _)
