import random
import tensorflow as tf


def i2b(x):
    s = f"{x:08b}"
    return [int(c) for c in reversed(s)]


def b2i(s):
    return int("".join([str(_) for _ in reversed(s)]), 2)


def ii2v(x1, x2):
    x1 = i2b(x1)
    x2 = i2b(x2)
    return tf.constant([[c1, c2] for c1, c2 in zip(x1, x2)], dtype=tf.float32)


def i2v(x):
    x = i2b(x)
    return tf.constant([c for c in x], dtype=tf.float32)


def f():
    while True:
        a = random.randint(0, 127)
        b = random.randint(0, 127)
        c = a + b
        yield ii2v(a, b), i2v(c)


def test(model, a, b):
    input = ii2v(a, b)
    input = tf.expand_dims(input, axis=0)
    logits = model(input)
    logits = tf.squeeze(logits)
    logits = tf.nn.softmax(logits, axis=1)
    result = tf.math.argmax(logits, axis=1)
    result = result.numpy().tolist()
    c = b2i(result)
    return c


epochs = 100
num = 128
bs = 32

dataset = tf.data.Dataset.from_generator(
    f,
    output_shapes=(tf.TensorShape((8, 2)), tf.TensorShape((8,))),
    output_types=(tf.float32, tf.float32)
).take(num * bs).batch(bs)

model = tf.keras.Sequential([
    tf.keras.layers.SimpleRNN(16, return_sequences=True, input_shape=(8, 2)),
    tf.keras.layers.Dense(2),
])
loss = tf.losses.SparseCategoricalCrossentropy(from_logits=True)
optimizer = tf.keras.optimizers.SGD()
model.compile(optimizer=optimizer, loss=loss)
model.fit(dataset, epochs=epochs)

print(f"5 + 8 = {test(model, 5, 8)}")

