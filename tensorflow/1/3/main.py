import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import xlrd


# step1: read in data from .xls file
DATA_FILE = 'fire_theft.xls'
book = xlrd.open_workbook(DATA_FILE, encoding_override='utf-8')
sheet = book.sheet_by_index(0)
data = np.asarray([sheet.row_values(i) for i in range(1, sheet.nrows)])
n_samples = sheet.nrows - 1


# step2: create placeholders for input X (number of file) and label Y (number of theft)
X = tf.placeholder(tf.float32, name='X')
Y = tf.placeholder(tf.float32, name='Y')


# step3: create weight and bias, initialized to 0
w = tf.Variable(0.0, name='weights')
b = tf.Variable(0.0, name='bias')


# step4: build model to predict Y
Y_predicted = X * w + b


# step5: use the square error as the loss function
loss = tf.square(Y - Y_predicted, name='loss')


# step6: using gradient descent with learning rate of 0.01 to minimize loss
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(loss)


with tf.Session() as sess:
    # step7: train
    with tf.summary.FileWriter('graphs', sess.graph) as writer:
        sess.run(tf.global_variables_initializer())
        for i in range(100):
            total_loss = 0
            for x, y in data:
                _, l = sess.run([optimizer, loss], feed_dict={X: x, Y:y})
                total_loss += l
            print('Epoch {0}: {1}'.format(i, total_loss / n_samples))
    
    # step8: plot
    X, Y = data.T[0], data.T[1]
    w, b = sess.run([w, b])
    plt.plot(X, Y, 'bo', label='Real data')
    plt.plot(X, X * w + b, 'r', label='Predicted data')
    plt.legend()
    plt.show()
    