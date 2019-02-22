import tensorflow as tf

a = tf.constant(2, name="a")
b = tf.constant(3, name="b")
x = tf.add(a, b, name="add")

with tf.Session() as sess:
    with tf.summary.FileWriter("./graphs", sess.graph) as writer:
        print(sess.run(x))