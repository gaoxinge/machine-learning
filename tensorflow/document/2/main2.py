# tensorboard --logdir=logs
import os
import tensorflow as tf


def scale(image, label):
    image = tf.cast(image, tf.float32)
    image /= 255
    return image, label


def decay(epoch):
    if epoch < 3:
        return 1e-3
    elif epoch < 7:
        return 1e-4
    else:
        return 1e-5


class PrintLR(tf.keras.callbacks.Callback):

    def on_epoch_end(self, epoch, logs=None):
        print('\nLearning rate for epoch {} is {}'.format(epoch + 1, model.optimizer.lr.numpy()))


strategy = tf.distribute.MirroredStrategy()
print('Number of devices: {}'.format(strategy.num_replicas_in_sync))


BUFFER_SIZE = 10000
BATCH_SIZE_PER_REPLICA = 64
BATCH_SIZE = BATCH_SIZE_PER_REPLICA * strategy.num_replicas_in_sync
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))\
    .map(scale)\
    .cache()\
    .shuffle(BUFFER_SIZE)\
    .batch(BATCH_SIZE)
eval_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))\
    .map(scale)\
    .batch(BATCH_SIZE)
print(x_train.shape[0] / BATCH_SIZE)


with strategy.scope():
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, 3, activation='relu', input_shape=(28, 28, 1)),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(10)
    ])

    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  optimizer=tf.keras.optimizers.Adam(),
                  metrics=['accuracy'])

    checkpoint_dir = './training_checkpoints'
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")
    callbacks = [
        tf.keras.callbacks.TensorBoard(log_dir='./logs'),
        tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_prefix, save_weights_only=True),
        tf.keras.callbacks.LearningRateScheduler(decay),
        PrintLR()
    ]

    EPOCHS = 12
    model.fit(train_dataset, epochs=EPOCHS, callbacks=callbacks)
    model.evaluate(eval_dataset)

