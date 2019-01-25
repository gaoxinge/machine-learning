import tf_api as tf

tf.Graph().as_default()

a = tf.Constant(15)
b = tf.Constant(5)
prod = tf.multiply(a, b)
sum = tf.add(a, b)
res = tf.divide(prod, sum)

session = tf.Session()
out = session.run(res)
print(out)
