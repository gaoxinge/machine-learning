import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.backend import get_session
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import ZeroPadding2D, Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import sparse_categorical_crossentropy
from tensorflow.keras.metrics import sparse_categorical_accuracy


(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1).astype(np.float32) / 255
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1).astype(np.float32) / 255


model = Sequential()
model.add(ZeroPadding2D(2, input_shape=(28, 28, 1)))
model.add(Conv2D(filters=6, kernel_size=(5, 5), activation=tf.nn.relu))
model.add(MaxPooling2D())
model.add(Conv2D(filters=16, kernel_size=(5, 5), activation=tf.nn.relu))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(units=120, activation=tf.nn.relu))
model.add(Dense(units=84, activation=tf.nn.relu))
model.add(Dense(units=10, activation=tf.nn.softmax))
model.compile(optimizer=Adam(0.001), loss=sparse_categorical_crossentropy, metrics=[sparse_categorical_accuracy])
model.fit(x=x_train, y=y_train, epochs=10)
loss, acc = model.evaluate(x=x_test, y=y_test)
print("loss: %s, acc: %s" % (loss, acc))


def process(image):
    return X.astype(np.float32) / 255


def deprocess(image):
    image = image * 255
    return np.clip(image, 0, 255).astype(np.uint8)


def clip(image):
    return np.clip(image, 0, 1)


l2_reg = 1e-3
learning_rate = 25
epochs = 200
epoch0 = 10
epoch1 = 10
target = 9

sess = get_session()
X = 255 * np.random.rand(28, 28, 1)
X = process(X)
X = X[None]
for _ in range(epochs):
    Y = tf.convert_to_tensor(X)
    with tf.GradientTape() as tape:
        tape.watch(Y)
        loss = model(Y)[0, target] - l2_reg * tf.nn.l2_loss(Y)
    dY = tape.gradient(loss, Y)
    loss, dX = sess.run([loss, dY])
    X += dX[0] * learning_rate
    X = clip(X)

    if _ % epoch0 == 0:
        print(_, loss)

    if _ % epoch1 == 0:
        Y = deprocess(X[0])
        plt.imshow(Y[:, :, 0], cmap='gray', vmin=0, vmax=255)
        plt.axis("off")
        plt.savefig("image/%s.jpg" % _)
