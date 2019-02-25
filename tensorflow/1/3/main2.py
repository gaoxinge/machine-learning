import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import time


learning_rate = 0.01
batch_size = 128
n_epochs = 30


# step1: read data
mnist = input_data.read_data_sets('./mnist', one_hot=True)


# step2: create placeholders
X = tf.placeholder(tf.float32, [batch_size, 784], name='X')
Y = tf.placeholder(tf.float32, [batch_size, 10], name='Y')


# step3: create variables
w = tf.Variable(tf.random.normal(shape=[784, 10], stddev=0.01), name='w')
b = tf.Variable(tf.zeros([1, 10]), name='b')


# step4: create logits
logits = tf.matmul(X, w) + b


# step5: create loss
entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=Y, name='loss')
loss = tf.reduce_mean(entropy)


# step6: create optimizer
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)


# step7: create accuracy
preds = tf.nn.softmax(logits)
correct_preds = tf.equal(tf.argmax(preds, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_sum(tf.cast(correct_preds, tf.float32))


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    with tf.summary.FileWriter('graphs', sess.graph) as writer:
        # step8: train
        start_time = time.time()
        n_batches = mnist.train.num_examples // batch_size
        for i in range(n_epochs):
            total_loss = 0
            for _ in range(n_batches):
                X_batch, Y_batch = mnist.train.next_batch(batch_size)
                _, loss_batch = sess.run([optimizer, loss], feed_dict={X: X_batch, Y: Y_batch})
                total_loss += loss_batch
            print('Average loss epoch {0}: {1}'.format(i, total_loss / n_batches))
        print('Total time: {0} seconds'.format(time.time() - start_time))
        print('Optimization Finished!')
        
        # step9: test
        n_batches = mnist.test.num_examples // batch_size
        total_correct_preds = 0
        for i in range(n_batches):
            X_batch, Y_batch = mnist.test.next_batch(batch_size)
            accuracy_batch = sess.run([accuracy], feed_dict={X: X_batch, Y: Y_batch})
            total_correct_preds += accuracy_batch[0]
        print('Accuracy {0}'.format(total_correct_preds / mnist.test.num_examples))