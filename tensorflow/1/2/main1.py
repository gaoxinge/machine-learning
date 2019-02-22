import tensorflow as tf

a = tf.constant(2)
b = tf.constant(3)
x = tf.add(a, b)

with tf.Session() as sess:
    with tf.summary.FileWriter('./graphs', sess.graph) as writer:
        print(sess.run(x))
        
# $ tensorboard --logdir="./graphs" --port 6006