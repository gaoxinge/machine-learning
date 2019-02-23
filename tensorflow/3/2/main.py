import tensorflow as tf

a = tf.constant(1)
b = tf.constant(2)
c = tf.multiply(a, b)
d = tf.Variable(1)
e = tf.placeholder(tf.int32)

print(tf.get_default_session())

with tf.Session().as_default() as sess:
    print(tf.get_default_session() is sess)
    print(tf.get_default_session())
    sess.run(d.initializer)
    print(a.eval(), b.eval(), c.eval(), d.eval(), e.eval(feed_dict={e:1}))

print(tf.get_default_session())
