import tensorflow as tf

# test read
with tf.gfile.GFile("20000.jpg", "rb") as fid:
    a = fid.read()

with tf.gfile.FastGFile("20000.jpg", "rb") as fid:
    b = fid.read()

with open("20000.jpg", "rb") as fid:
    c = fid.read()

print(a == b == c)

# test readlines
with tf.gfile.GFile("tmp.txt") as fid:
    a = fid.readlines()

with tf.gfile.FastGFile("tmp.txt") as fid:
    b = fid.readlines()

with open("tmp.txt") as fid:
    c = fid.readlines()

print(a == b == c)
