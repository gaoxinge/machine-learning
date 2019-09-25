import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import ZeroPadding2D, Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import sparse_categorical_crossentropy
from tensorflow.keras.metrics import sparse_categorical_accuracy


x_train = np.zeros((5000, 224, 224, 3), dtype=np.uint8)
y_train = np.zeros((5000, 1), dtype=np.uint8)
x_test = np.zeros((1000, 224, 224, 3), dtype=np.uint8)
y_test = np.zeros((1000, 1), dtype=np.uint8)


model = Sequential()
model.add(Conv2D(filters=96, kernel_size=(11, 11), strides=(4, 4), input_shape=(224, 224, 3), activation=tf.nn.relu))
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
