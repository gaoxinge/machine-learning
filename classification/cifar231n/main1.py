import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import sparse_categorical_crossentropy
from tensorflow.keras.metrics import sparse_categorical_accuracy
from cs231n.data_utils import load_CIFAR10


x_train, y_train, x_test, y_test = load_CIFAR10("cs231n/datasets/cifar-10-batches-py/")
x_train = x_train.reshape(x_train.shape[0], 32 * 32 * 3).astype(np.float32) / 255
x_test = x_test.reshape(x_test.shape[0], 32 * 32 * 3).astype(np.float32) / 255


model = Sequential()
model.add(Dense(10, input_dim=3072, activation=tf.nn.softmax))
model.compile(optimizer=Adam(0.001), loss=sparse_categorical_crossentropy, metrics=[sparse_categorical_accuracy])
model.fit(x=x_train, y=y_train, epochs=10)
model.evaluate(x=x_test, y=y_test)
