# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np


# utility functions to handle binary number
binary_dim = 8
largest_number = pow(2, binary_dim)
binary = np.unpackbits(np.array([range(largest_number)], dtype=np.uint8).T, axis=1)
int2binary = {i: binary[i] for i in range(largest_number)}


def binary_generation(numbers, reverse=False):
    binary_x = np.array([int2binary[num] for num in numbers], dtype=np.uint8)
    if reverse:
        binary_x = np.fliplr(binary_x)
    return binary_x


def batch_generation(batch_size):
    n1 = np.random.randint(0, largest_number // 2, batch_size)
    n2 = np.random.randint(0, largest_number // 2, batch_size)
    add = n1 + n2

    binary_n1 = binary_generation(n1, True)
    binary_n2 = binary_generation(n2, True)
    batch_x = np.dstack((binary_n1, binary_n2))
    batch_y = binary_generation(add, True)

    return batch_x, batch_y, n1, n2, add


def binary2int(binary_array):
    out = 0
    for index, x in enumerate(reversed(binary_array)):
        out += x * pow(2, index)
    return out


# basic rnn model
batch_size = 64
lstm_size = 20

x = tf.placeholder(tf.float32, [batch_size, binary_dim, 2])
y = tf.placeholder(tf.float32, [batch_size, binary_dim])

cell = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.BasicLSTMCell(lstm_size) for _ in range(2)])
initial_state = cell.zero_state(batch_size, tf.float32)
outputs, final_state = tf.nn.dynamic_rnn(cell, x, initial_state=initial_state)
# [batch_size, binary_dim, lstm_size] => [batch_size * binary_dim, lstm_size]
outputs = tf.reshape(outputs, [-1, lstm_size])

w = tf.Variable(tf.zeros([lstm_size, 1]))
b = tf.Variable(tf.zeros([1]))
y_ = tf.sigmoid(tf.matmul(outputs, w) + b)
# [batch_size * binary_dim, 1] => [batch_size, binary_dim]
y_ = tf.reshape(y_, [-1, binary_dim])

loss = tf.losses.mean_squared_error(y, y_)
optimizer = tf.train.AdamOptimizer().minimize(loss)


# train and test
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for _ in range(2000):
        input_x, input_y, _, _, _ = batch_generation(batch_size)
        sess.run([optimizer, loss], feed_dict={x: input_x, y: input_y})

    val_x, val_y, n1, n2, add = batch_generation(batch_size)
    result = sess.run(y_, feed_dict={x: val_x, y: val_y})

    # round: 四舍五入
    # fliplr: 反转
    # astype: 浮点数转化为整数
    result = np.fliplr(np.round(result)).astype(np.int32)

    for b_x, b_p, a, b, add in zip(np.fliplr(val_x), result, n1, n2, add):
        print('{}: {}'.format(b_x[:, 0], a))
        print('{}: {}'.format(b_x[:, 1], b))
        print('{}: {}\n'.format(b_p, binary2int(b_p)))
