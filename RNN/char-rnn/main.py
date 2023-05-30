import os
import tqdm
import tensorflow as tf


path_to_file = tf.keras.utils.get_file(
    "shakespeare.txt",
    "https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt"
)

with open(path_to_file, "r", encoding="utf-8") as f:
    text = f.read()
vocab = sorted(set(text))

ids_from_chars = tf.keras.layers.StringLookup(vocabulary=list(vocab), mask_token=None)
chars_from_ids = tf.keras.layers.StringLookup(vocabulary=ids_from_chars.get_vocabulary(), invert=True, mask_token=None)

print("=" * 80)
print(len(vocab), vocab)  # len = 65
print(len(ids_from_chars.get_vocabulary()), ids_from_chars.get_vocabulary())  # len = 66, [UNK]

print("=" * 80)
print(ids_from_chars(["a", "b", "c"]))
print(ids_from_chars([["a", "b", "c"], ["d", "e", "f"]]))
print(ids_from_chars(["高"]))
print(chars_from_ids([40, 41, 42]))
print(chars_from_ids([[40, 41, 42], [43, 44, 45]]))
print(chars_from_ids([0, 1000]))

print("=" * 80)
print(tf.strings.unicode_split("abc", "UTF-8"))
print(tf.strings.unicode_split(["abc", "def"], "UTF-8"))
print(tf.strings.join(["a", "b", "c"]))
print(tf.strings.join([["a", "b", "c"]]))
print(tf.strings.join([["a", "b", "c"], ["d", "e", "f"]]))
print(tf.strings.join(tf.RaggedTensor.from_tensor([["a", "b", "c"], ["d", "e", "f"]])))


def split_input_target(sequence):
    input_text = sequence[:-1]
    target_text = sequence[1:]
    return input_text, target_text


dataset = tf.data.Dataset.from_tensor_slices(ids_from_chars(tf.strings.unicode_split(text, "UTF-8")))\
    .batch(100 + 1, drop_remainder=True)\
    .map(split_input_target)\
    .shuffle(10000)\
    .batch(128, drop_remainder=True)\
    .prefetch(tf.data.experimental.AUTOTUNE)

print("=" * 80)
for data, label in dataset.take(1):
    print(data.shape, label.shape)


class MyModel(tf.keras.Model):

    def __init__(self, vocab_size, embedding_dim, rnn_units):
        super().__init__(self)
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = tf.keras.layers.GRU(
            rnn_units,
            return_sequences=True,
            return_state=True
        )
        self.dense = tf.keras.layers.Dense(vocab_size)

    def call(self, inputs, states=None, return_state=False, training=False):
        x = self.embedding(inputs, training=training)
        if states is None:
            states = self.gru.get_initial_state(x)
        x, states = self.gru(x, initial_state=states, training=training)
        x = self.dense(x, training=training)
        return (x, states) if return_state else x


model = MyModel(vocab_size=len(ids_from_chars.get_vocabulary()), embedding_dim=256, rnn_units=512)
loss = tf.losses.SparseCategoricalCrossentropy(from_logits=True)
optimizer = tf.keras.optimizers.Adam()
train_loss = tf.keras.metrics.Mean(name="train_loss")
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name="train_accuracy")
checkpoint_dir = "./training_checkpoints"
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(model=model)


@tf.function
def step_train(data, label):
    with tf.GradientTape() as tape:
        result = model(data, training=True)
        loss_ = loss(label, result)
        gradients = tape.gradient(loss_, model.trainable_variables)
        gradients, _ = tf.clip_by_global_norm(gradients, 5)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        train_loss(loss_)
        train_accuracy(label, result)


print("=" * 80)
for epoch in range(20):
    train_loss.reset_state()
    train_accuracy.reset_state()

    for data, label in tqdm.tqdm(dataset):
        step_train(data, label)

    checkpoint.save(checkpoint_prefix)

    print(
        f'Epoch {epoch + 1}, '
        f'Loss: {train_loss.result()}, '
        f'Accuracy: {train_accuracy.result() * 100}'
    )


print("=" * 80)
checkpoint.restore(os.path.join(checkpoint_dir, "ckpt-20"))
next_chars = tf.constant(["ROMEO:"])
states = None
result = [next_chars]
for _ in tqdm.tqdm(range(1000)):
    input_chars = tf.strings.unicode_split(next_chars, "UTF-8")
    # like dataset with (64, 100), shape = (1, len(input_chars))
    input_ids = ids_from_chars(input_chars).to_tensor()
    # shape1 = (1, len(input_chars), 66), shape2 = (1, 1024)
    predicted_logits, states = model(inputs=input_ids, states=states, return_state=True)
    predicted_logits = predicted_logits[:, -1, :]  # the last char, shape = (1, 66)
    predicted_ids = tf.random.categorical(predicted_logits, num_samples=1)  # sample by softmax, shape = (1, 1)
    predicted_ids = tf.RaggedTensor.from_tensor(predicted_ids)
    predicted_chars = chars_from_ids(predicted_ids)
    next_chars = tf.strings.join(predicted_chars)
    result.append(next_chars)

print("=" * 80)
result = tf.strings.join(result)
print(result[0].numpy().decode("utf-8"))

