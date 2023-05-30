"""
- https://github.com/karpathy/char-rnn
- https://cs.stanford.edu/people/karpathy/char-rnn/
- https://github.com/NELSONZHAO/zhihu/tree/master/anna_lstm
- https://zhuanlan.zhihu.com/p/27087310
- https://github.com/hzy46/Char-RNN-TensorFlow
"""
import os
import tqdm
import tensorflow as tf


class LSTMModel(tf.keras.Model):

    def __init__(self, vocab_size, embedding_dim, rnn_units):
        super().__init__(self)
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.lstm = tf.keras.layers.LSTM(
            rnn_units,
            return_sequences=True,
            return_state=True
        )
        self.dense = tf.keras.layers.Dense(vocab_size)

    def call(self, inputs, states=None, return_state=False, training=False):
        x = self.embedding(inputs, training=training)
        if states is None:
            states = self.lstm.get_initial_state(x)
        x, states1, states2 = self.lstm(x, initial_state=states, training=training)
        states = [states1, states2]
        x = self.dense(x, training=training)
        return (x, states) if return_state else x


class GRUModel(tf.keras.Model):

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


class Manager:

    def __init__(self, model, ids_from_chars, chars_from_ids):
        self.model = model
        self.ids_from_chars = ids_from_chars
        self.chars_from_ids = chars_from_ids
        self.checkpoint = tf.train.Checkpoint(model=self.model)

    def train(self, text, checkpoint_prefix, epochs, seq_num, batch_size):
        def split_input_target(sequence):
            input_text = sequence[:-1]
            target_text = sequence[1:]
            return input_text, target_text

        @tf.function
        def step_train(data, label):
            with tf.GradientTape() as tape:
                result = self.model(data, training=True)
                loss_ = loss(label, result)
                gradients = tape.gradient(loss_, self.model.trainable_variables)
                gradients, _ = tf.clip_by_global_norm(gradients, 5)
                optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
                train_loss(loss_)
                train_accuracy(label, result)

        dataset = tf.data.Dataset \
            .from_tensor_slices(self.ids_from_chars(tf.strings.unicode_split(text, "UTF-8"))) \
            .batch(seq_num + 1, drop_remainder=True) \
            .map(split_input_target) \
            .shuffle(10000) \
            .batch(batch_size, drop_remainder=True) \
            .prefetch(tf.data.experimental.AUTOTUNE)

        loss = tf.losses.SparseCategoricalCrossentropy(from_logits=True)
        optimizer = tf.keras.optimizers.Adam()
        train_loss = tf.keras.metrics.Mean(name="train_loss")
        train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name="train_accuracy")

        for epoch in range(epochs):
            train_loss.reset_state()
            train_accuracy.reset_state()

            for data, label in tqdm.tqdm(dataset):
                step_train(data, label)

            self.checkpoint.save(checkpoint_prefix)

            print(
                f'Epoch {epoch + 1}, '
                f'Loss: {train_loss.result()}, '
                f'Accuracy: {train_accuracy.result() * 100}'
            )

    def sample(self, checkpoint_path, char_start, char_length):
        self.checkpoint.restore(checkpoint_path)
        next_chars, states = tf.constant([char_start]), None
        result = [next_chars]
        for _ in tqdm.tqdm(range(char_length)):
            input_chars = tf.strings.unicode_split(next_chars, "UTF-8")
            input_ids = self.ids_from_chars(input_chars).to_tensor()
            predicted_logits, states = self.model(inputs=input_ids, states=states, return_state=True)
            predicted_logits = predicted_logits[:, -1, :]
            predicted_ids = tf.random.categorical(predicted_logits, num_samples=1)
            predicted_ids = tf.RaggedTensor.from_tensor(predicted_ids)
            predicted_chars = self.chars_from_ids(predicted_ids)
            next_chars = tf.strings.join(predicted_chars)
            result.append(next_chars)
        result = tf.strings.join(result)
        return result[0].numpy().decode("utf-8")


#########
# linux #
#########
def init_linux():
    path_to_file = tf.keras.utils.get_file(
        "linux.txt",
        "https://cs.stanford.edu/people/karpathy/char-rnn/linux_input.txt"
    )
    with open(path_to_file, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()
    vocab = sorted(set(text))
    ids_from_chars = tf.keras.layers.StringLookup(vocabulary=list(vocab), mask_token=None)
    chars_from_ids = tf.keras.layers.StringLookup(
        vocabulary=ids_from_chars.get_vocabulary(),
        invert=True,
        mask_token=None
    )
    model = GRUModel(vocab_size=len(ids_from_chars.get_vocabulary()), embedding_dim=256, rnn_units=512)
    checkpoint_dir = "./training_checkpoints_linux"
    manager = Manager(model, ids_from_chars, chars_from_ids)
    return text, manager, checkpoint_dir


def train_linux():
    text, manager, checkpoint_dir = init_linux()
    manager.train(text, os.path.join(checkpoint_dir, "ckpt"), 20, 100, 128)


def sample_linux():
    text, manager, checkpoint_dir = init_linux()
    result = manager.sample(os.path.join(checkpoint_dir, "ckpt-20"), "#", 3000)
    print(result)


###############
# shakespeare #
###############
def init_shakespeare():
    path_to_file = tf.keras.utils.get_file(
        "shakespeare.txt",
        "https://cs.stanford.edu/people/karpathy/char-rnn/shakespeare_input.txt"
    )
    with open(path_to_file, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()
    vocab = sorted(set(text))
    ids_from_chars = tf.keras.layers.StringLookup(vocabulary=list(vocab), mask_token=None)
    chars_from_ids = tf.keras.layers.StringLookup(
        vocabulary=ids_from_chars.get_vocabulary(),
        invert=True,
        mask_token=None
    )
    model = GRUModel(vocab_size=len(ids_from_chars.get_vocabulary()), embedding_dim=256, rnn_units=512)
    checkpoint_dir = "./training_checkpoints_shakespeare"
    manager = Manager(model, ids_from_chars, chars_from_ids)
    return text, manager, checkpoint_dir


def train_shakespeare():
    text, manager, checkpoint_dir = init_shakespeare()
    manager.train(text, os.path.join(checkpoint_dir, "ckpt"), 20, 100, 128)


def sample_shakespeare():
    text, manager, checkpoint_dir = init_shakespeare()
    result = manager.sample(os.path.join(checkpoint_dir, "ckpt-20"), "DEMO:", 1000)
    print(result)


############
# warpeace #
############
def init_warpeace():
    path_to_file = tf.keras.utils.get_file(
        "warpeace.txt",
        "https://cs.stanford.edu/people/karpathy/char-rnn/warpeace_input.txt"
    )
    with open(path_to_file, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()
    vocab = sorted(set(text))
    ids_from_chars = tf.keras.layers.StringLookup(vocabulary=list(vocab), mask_token=None)
    chars_from_ids = tf.keras.layers.StringLookup(
        vocabulary=ids_from_chars.get_vocabulary(),
        invert=True,
        mask_token=None
    )
    model = GRUModel(vocab_size=len(ids_from_chars.get_vocabulary()), embedding_dim=256, rnn_units=512)
    checkpoint_dir = "./training_checkpoints_warpeace"
    manager = Manager(model, ids_from_chars, chars_from_ids)
    return text, manager, checkpoint_dir


def train_warpeace():
    text, manager, checkpoint_dir = init_warpeace()
    manager.train(text, os.path.join(checkpoint_dir, "ckpt"), 20, 100, 128)


def sample_warpeace():
    text, manager, checkpoint_dir = init_warpeace()
    result = manager.sample(os.path.join(checkpoint_dir, "ckpt-20"), "\"", 3000)
    print(result)


#######
# jay #
#######
def init_jay():
    path_to_file = tf.keras.utils.get_file(
        "jay.txt",
        "https://raw.githubusercontent.com/hzy46/Char-RNN-TensorFlow/master/data/jay.txt"
    )
    with open(path_to_file, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()
    vocab = sorted(set(text))
    ids_from_chars = tf.keras.layers.StringLookup(vocabulary=list(vocab), mask_token=None)
    chars_from_ids = tf.keras.layers.StringLookup(
        vocabulary=ids_from_chars.get_vocabulary(),
        invert=True,
        mask_token=None
    )
    model = GRUModel(vocab_size=len(ids_from_chars.get_vocabulary()), embedding_dim=256, rnn_units=512)
    checkpoint_dir = "./training_checkpoints_jay"
    manager = Manager(model, ids_from_chars, chars_from_ids)
    return text, manager, checkpoint_dir


def train_jay():
    text, manager, checkpoint_dir = init_jay()
    manager.train(text, os.path.join(checkpoint_dir, "ckpt"), 100, 100, 128)


def sample_jay():
    text, manager, checkpoint_dir = init_jay()
    result = manager.sample(os.path.join(checkpoint_dir, "ckpt-100"), "作者", 100)
    print(result)


##########
# poetry #
##########
def init_poetry():
    path_to_file = tf.keras.utils.get_file(
        "poetry.txt",
        "https://raw.githubusercontent.com/hzy46/Char-RNN-TensorFlow/master/data/poetry.txt"
    )
    with open(path_to_file, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()
    vocab = sorted(set(text))
    ids_from_chars = tf.keras.layers.StringLookup(vocabulary=list(vocab), mask_token=None)
    chars_from_ids = tf.keras.layers.StringLookup(
        vocabulary=ids_from_chars.get_vocabulary(),
        invert=True,
        mask_token=None
    )
    model = GRUModel(vocab_size=len(ids_from_chars.get_vocabulary()), embedding_dim=256, rnn_units=512)
    checkpoint_dir = "./training_checkpoints_poetry"
    manager = Manager(model, ids_from_chars, chars_from_ids)
    return text, manager, checkpoint_dir


def train_poetry():
    text, manager, checkpoint_dir = init_poetry()
    manager.train(text, os.path.join(checkpoint_dir, "ckpt"), 100, 100, 128)


def sample_poetry():
    text, manager, checkpoint_dir = init_poetry()
    result = manager.sample(os.path.join(checkpoint_dir, "ckpt-100"), "高", 100)
    print(result)


if __name__ == "__main__":
    train_linux()
    sample_linux()

    # train_shakespeare()
    # sample_shakespeare()

    # train_warpeace()
    # sample_warpeace()

    # train_jay()
    # sample_jay()

    # train_poetry()
    # sample_poetry()

