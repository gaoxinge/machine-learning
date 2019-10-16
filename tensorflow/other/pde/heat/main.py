import numpy as np
import tensorflow as tf
from PIL import Image
tf.enable_eager_execution()


def make_kernel(a):
    a = tf.constant(a, dtype=tf.float32)
    a = tf.reshape(a, [3, 3, 1, 1])
    return a


def simple_conv(x, k):
    x = tf.expand_dims(tf.expand_dims(x, 0), -1)
    y = tf.nn.conv2d(input=x, filter=k, strides=[1, 1, 1, 1], padding="SAME")
    return y[0, :, :, 0]


def laplace(x):
    laplace_k = make_kernel([[0.5, 1.0, 0.5],
                             [1.0, -6.0, 1.0],
                             [0.5, 1.0, 0.5]])
    return simple_conv(x, laplace_k)


def save(a, name, rng):
    a = (a - rng[0]) / (rng[1] - rng[0]) * 255
    a = np.clip(a, 0, 255).astype(np.uint8)
    image = Image.fromarray(a)
    image.save(name)


N = 500
h = 0.03
steps = 1000
step0 = 100

U = np.zeros([N, N], dtype=np.float32)
for n in range(40):
    a, b = np.random.randint(0, N, 2)
    U[a, b] = 255 * np.random.uniform()
U = tf.Variable(U)

for _ in range(steps):
    U = U + laplace(U) * h
    if _ % step0 == 0:
       save(U.numpy(), "image/%s.jpg" % _, [-0.1, 0.1])
