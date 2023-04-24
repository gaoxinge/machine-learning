import tensorflow as tf

# tf.enable_eager_execution()  # tf 2.10.1 not need

x = tf.constant(3.0)
for _ in range(10000):
    if _ % 100 == 0:
        print(_)
    with tf.GradientTape() as g:
        g.watch(x)
        y = x * x
    dy_dx = g.gradient(y, x)
    x -= dy_dx * 0.001
print(x.numpy())
