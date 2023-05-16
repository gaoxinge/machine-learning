import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import ZeroPadding2D, Conv2D, MaxPooling2D, Flatten, Dense, BatchNormalization, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import sparse_categorical_crossentropy
from tensorflow.keras.metrics import sparse_categorical_accuracy


# mnist
def scale(image, label):
    image = tf.cast(image, tf.float32)
    image /= 255
    return image, label


(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
ds_train = tf.data.Dataset.from_tensor_slices((x_train, y_train)).map(scale).shuffle(10000).batch(32)
ds_test = tf.data.Dataset.from_tensor_slices((x_test, y_test)).map(scale).batch(32)

model = Sequential()
model.add(Conv2D(filters=96, kernel_size=(11, 11), strides=(4, 4), padding="same", input_shape=(28, 28, 1), activation=tf.nn.relu))
model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding="same"))
model.add(BatchNormalization())
model.add(Conv2D(filters=256, kernel_size=(5, 5), padding="same", activation=tf.nn.relu))
model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding="same"))
model.add(BatchNormalization())
model.add(Conv2D(filters=384, kernel_size=(3, 3), padding="same", activation=tf.nn.relu))
model.add(Conv2D(filters=384, kernel_size=(3, 3), padding="same", activation=tf.nn.relu))
model.add(Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation=tf.nn.relu))
model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding="same"))
model.add(Flatten())
model.add(Dense(units=4096, activation=tf.nn.relu))
model.add(Dropout(0.5))
model.add(Dense(units=4096, activation=tf.nn.relu))
model.add(Dropout(0.5))
model.add(Dense(units=10, activation=tf.nn.softmax))
model.compile(optimizer=Adam(0.001), loss=sparse_categorical_crossentropy, metrics=[sparse_categorical_accuracy])
model.summary()
model.fit(ds_train, epochs=10)
loss, acc = model.evaluate(ds_test, verbose=0)
print(f"loss: {loss}, acc: {acc}")


# mnist fashion
def scale(image, label):
    image = tf.cast(image, tf.float32)
    image /= 255
    return image, label


(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
ds_train = tf.data.Dataset.from_tensor_slices((x_train, y_train)).map(scale).shuffle(10000).batch(32)
ds_test = tf.data.Dataset.from_tensor_slices((x_test, y_test)).map(scale).batch(32)

model = Sequential()
model.add(Conv2D(filters=96, kernel_size=(11, 11), strides=(4, 4), padding="same", input_shape=(28, 28, 1), activation=tf.nn.relu))
model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding="same"))
model.add(BatchNormalization())
model.add(Conv2D(filters=256, kernel_size=(5, 5), padding="same", activation=tf.nn.relu))
model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding="same"))
model.add(BatchNormalization())
model.add(Conv2D(filters=384, kernel_size=(3, 3), padding="same", activation=tf.nn.relu))
model.add(Conv2D(filters=384, kernel_size=(3, 3), padding="same", activation=tf.nn.relu))
model.add(Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation=tf.nn.relu))
model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding="same"))
model.add(Flatten())
model.add(Dense(units=4096, activation=tf.nn.relu))
model.add(Dropout(0.5))
model.add(Dense(units=4096, activation=tf.nn.relu))
model.add(Dropout(0.5))
model.add(Dense(units=10, activation=tf.nn.softmax))
model.compile(optimizer=Adam(0.001), loss=sparse_categorical_crossentropy, metrics=[sparse_categorical_accuracy])
model.summary()
model.fit(ds_train, epochs=10)
loss, acc = model.evaluate(ds_test, verbose=0)
print(f"loss: {loss}, acc: {acc}")


# cifar10
def scale(image, label):
    image = tf.cast(image, tf.float32)
    image /= 255
    return image, label


(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
ds_train = tf.data.Dataset.from_tensor_slices((x_train, y_train)).map(scale).shuffle(10000).batch(32)
ds_test = tf.data.Dataset.from_tensor_slices((x_test, y_test)).map(scale).batch(32)

model = Sequential()
model.add(Conv2D(filters=96, kernel_size=(11, 11), strides=(4, 4), padding="same", input_shape=(32, 32, 3), activation=tf.nn.relu))
model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding="same"))
model.add(BatchNormalization())
model.add(Conv2D(filters=256, kernel_size=(5, 5), padding="same", activation=tf.nn.relu))
model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding="same"))
model.add(BatchNormalization())
model.add(Conv2D(filters=384, kernel_size=(3, 3), padding="same", activation=tf.nn.relu))
model.add(Conv2D(filters=384, kernel_size=(3, 3), padding="same", activation=tf.nn.relu))
model.add(Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation=tf.nn.relu))
model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding="same"))
model.add(Flatten())
model.add(Dense(units=4096, activation=tf.nn.relu))
model.add(Dropout(0.5))
model.add(Dense(units=4096, activation=tf.nn.relu))
model.add(Dropout(0.5))
model.add(Dense(units=10, activation=tf.nn.softmax))
model.compile(optimizer=Adam(0.001), loss=sparse_categorical_crossentropy, metrics=[sparse_categorical_accuracy])
model.summary()
model.fit(ds_train, epochs=10)
loss, acc = model.evaluate(ds_test, verbose=0)
print(f"loss: {loss}, acc: {acc}")

