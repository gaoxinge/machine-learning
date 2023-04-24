import numpy as np
import tensorflow as tf


with tf.compat.v1.Session() as sess:
    x = tf.constant(3.0)
    for _ in range(10000):
        if _ % 100 == 0:
            print(_)
        with tf.GradientTape() as g:
            g.watch(x)
            y = x * x
        dy_dx = g.gradient(y, x)
        x -= dy_dx * 0.001
    print(sess.run(x))
