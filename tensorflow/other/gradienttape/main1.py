import tensorflow as tf

# tf 2.10.1 not work
sess = tf.compat.v1.InteractiveSession()
x = tf.constant(3.0)
for _ in range(10000):
    if _ % 100 == 0:
        print(_)
    with tf.GradientTape() as g:
        g.watch(x)
        y = x * x
    dy_dx = g.gradient(y, x)
    x -= dy_dx * 0.001
print(x.eval())
sess.close()

