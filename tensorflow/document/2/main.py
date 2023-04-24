import tensorflow as tf
print(tf.__version__)
print(tf.config.experimental.list_physical_devices('GPU'))
print(tf.config.experimental.list_logical_devices('GPU'))

print("\nload data:")
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
print(x_train.shape)
print(x_train[:1])
print(x_train[0, :, :])
print(y_train.shape)
print(y_train[:1])
print(y_train[0])

print("\nbuild model:")
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10),
])
predictions = model(x_train[:1]).numpy()
print(predictions)
print(tf.nn.softmax(predictions).numpy())
predictions = model(x_train[0, :, :].reshape(1, 28, 28)).numpy()
print(predictions)
print(tf.nn.softmax(predictions).numpy())

print("\nbuild loss:")
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
print(loss_fn(y_train[:1], predictions).numpy())

print("\ncompile model:")
model.compile(optimizer="adam", loss=loss_fn, metrics=["accuracy"])

print("\ntrain and evaluate:")
model.fit(x_train, y_train, epochs=5)
model.evaluate(x_test, y_test, verbose=2)

print("\npredict:")
probability_model = tf.keras.Sequential([
    model,
    tf.keras.layers.Softmax()
])
p = probability_model(x_test[:5])
print(tf.math.argmax(p, 1))
print(y_test[:5])

