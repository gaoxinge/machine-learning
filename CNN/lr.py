import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import sparse_categorical_crossentropy
from tensorflow.keras.metrics import sparse_categorical_accuracy


# mnist
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = x_train.reshape(x_train.shape[0], 28 * 28).astype(np.float32) / 255  # float32 instead of float64
x_test = x_test.reshape(x_test.shape[0], 28 * 28).astype(np.float32) / 255  # float32 instead of float64

model = Sequential()
model.add(Dense(10, input_dim=784, activation=tf.nn.softmax))
model.compile(optimizer=Adam(0.001), loss=sparse_categorical_crossentropy, metrics=[sparse_categorical_accuracy])
model.fit(x=x_train, y=y_train, epochs=10)
loss, acc = model.evaluate(x=x_test, y=y_test, verbose=0)
print(f"loss: {loss}, acc: {acc}")


# mnist fashion
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
x_train = x_train.reshape(x_train.shape[0], 28 * 28).astype(np.float32) / 255
x_test = x_test.reshape(x_test.shape[0], 28 * 28).astype(np.float32) / 255

model = Sequential()
model.add(Dense(10, input_dim=784, activation=tf.nn.softmax))
model.compile(optimizer=Adam(0.001), loss=sparse_categorical_crossentropy, metrics=[sparse_categorical_accuracy])
model.fit(x=x_train, y=y_train, epochs=10)
loss, acc = model.evaluate(x=x_test, y=y_test, verbose=0)
print(f"loss: {loss}, acc: {acc}")


# cifar10
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
x_train = x_train.reshape(x_train.shape[0], 32 * 32 * 3).astype(np.float32) / 255
x_test = x_test.reshape(x_test.shape[0], 32 * 32 * 3).astype(np.float32) / 255

model = Sequential()
model.add(Dense(10, input_dim=3072, activation=tf.nn.softmax))
model.compile(optimizer=Adam(0.001), loss=sparse_categorical_crossentropy, metrics=[sparse_categorical_accuracy])
model.fit(x=x_train, y=y_train, epochs=10)
loss, acc = model.evaluate(x=x_test, y=y_test, verbose=0)
print(f"loss: {loss}, acc: {acc}")

