import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import ZeroPadding2D, Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import sparse_categorical_crossentropy
from tensorflow.keras.metrics import sparse_categorical_accuracy
from cs231n.data_utils import load_CIFAR10

config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)

x_train, y_train, x_test, y_test = load_CIFAR10("cs231n/datasets/cifar-10-batches-py/")
x_train = x_train.reshape(x_train.shape[0], 32, 32, 3).astype(np.float32) / 255
x_test = x_test.reshape(x_test.shape[0], 32, 32, 3).astype(np.float32) / 255


model = Sequential()
model.add(ZeroPadding2D(97, input_shape=(32, 32, 3)))
model.add(Conv2D(filters=96, kernel_size=(11, 11), strides=(4, 4), activation=tf.nn.relu))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Conv2D(filters=256, kernel_size=(5, 5), activation=tf.nn.relu))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Conv2D(filters=384, kernel_size=(3, 3), activation=tf.nn.relu))
model.add(Conv2D(filters=384, kernel_size=(3, 3), activation=tf.nn.relu))
model.add(Conv2D(filters=256, kernel_size=(3, 3), activation=tf.nn.relu))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Flatten())
model.add(Dense(units=4096, activation=tf.nn.relu))
model.add(Dropout(0.5))
model.add(Dense(units=4096, activation=tf.nn.relu))
model.add(Dropout(0.5))
model.add(Dense(units=10, activation=tf.nn.softmax))
model.compile(optimizer=Adam(0.001), loss=sparse_categorical_crossentropy, metrics=[sparse_categorical_accuracy])
model.fit(x=x_train, y=y_train, epochs=10)
model.evaluate(x=x_test, y=y_test)
