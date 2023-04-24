import tensorflow as tf

# test read
with tf.gfile.GFile("assets/20000.jpg", "rb") as fid:
    a = fid.read()

with tf.gfile.FastGFile("assets/20000.jpg", "rb") as fid:
    b = fid.read()

with open("assets/20000.jpg", "rb") as fid:
    c = fid.read()

print(a == b == c)

# test readlines
with tf.gfile.GFile("assets/tmp.txt") as fid:
    a = fid.readlines()

with tf.gfile.FastGFile("assets/tmp.txt") as fid:
    b = fid.readlines()

with open("assets/tmp.txt") as fid:
    c = fid.readlines()

print(a == b == c)
