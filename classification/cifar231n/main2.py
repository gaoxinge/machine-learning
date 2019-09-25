import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.losses import categorical_hinge
from tensorflow.keras.metrics import categorical_accuracy
from tensorflow.keras.utils import to_categorical
from cs231n.data_utils import load_CIFAR10


x_train, y_train, x_test, y_test = load_CIFAR10("cs231n/datasets/cifar-10-batches-py/")
x_train = x_train.reshape(x_train.shape[0], 32 * 32 * 3).astype(np.float32) / 255
x_test = x_test.reshape(x_test.shape[0], 32 * 32 * 3).astype(np.float32) / 255
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)


model = Sequential()
model.add(Dense(10, input_dim=3072, kernel_regularizer=l2(0.001)))
model.compile(optimizer=Adam(0.001), loss=categorical_hinge, metrics=[categorical_accuracy])
model.fit(x=x_train, y=y_train, epochs=10)
model.evaluate(x=x_test, y=y_test)
