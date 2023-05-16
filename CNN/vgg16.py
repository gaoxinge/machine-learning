import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import ZeroPadding2D, Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.applications import vgg16


model = Sequential()
model.add(ZeroPadding2D((1, 1), input_shape=(224, 224, 3)))
model.add(Conv2D(filters=64, kernel_size=(3, 3), activation=tf.nn.relu))
model.add(ZeroPadding2D((1, 1)))
model.add(Conv2D(filters=64, kernel_size=(3, 3), activation=tf.nn.relu))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

model.add(ZeroPadding2D((1, 1)))
model.add(Conv2D(filters=128, kernel_size=(3, 3), activation=tf.nn.relu))
model.add(ZeroPadding2D((1, 1)))
model.add(Conv2D(filters=128, kernel_size=(3, 3), activation=tf.nn.relu))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

model.add(ZeroPadding2D((1, 1)))
model.add(Conv2D(filters=256, kernel_size=(3, 3), activation=tf.nn.relu))
model.add(ZeroPadding2D((1, 1)))
model.add(Conv2D(filters=256, kernel_size=(3, 3), activation=tf.nn.relu))
model.add(ZeroPadding2D((1, 1)))
model.add(Conv2D(filters=256, kernel_size=(3, 3), activation=tf.nn.relu))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

model.add(ZeroPadding2D((1, 1)))
model.add(Conv2D(filters=512, kernel_size=(3, 3), activation=tf.nn.relu))
model.add(ZeroPadding2D((1, 1)))
model.add(Conv2D(filters=512, kernel_size=(3, 3), activation=tf.nn.relu))
model.add(ZeroPadding2D((1, 1)))
model.add(Conv2D(filters=512, kernel_size=(3, 3), activation=tf.nn.relu))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

model.add(ZeroPadding2D((1, 1)))
model.add(Conv2D(filters=512, kernel_size=(3, 3), activation=tf.nn.relu))
model.add(ZeroPadding2D((1, 1)))
model.add(Conv2D(filters=512, kernel_size=(3, 3), activation=tf.nn.relu))
model.add(ZeroPadding2D((1, 1)))
model.add(Conv2D(filters=512, kernel_size=(3, 3), activation=tf.nn.relu))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

model.add(Flatten())
model.add(Dense(units=4096, activation=tf.nn.relu))
model.add(Dense(units=4096, activation=tf.nn.relu))
model.add(Dense(units=1000, activation=tf.nn.softmax))
model.summary()


model = vgg16.VGG16()
model.summary()

