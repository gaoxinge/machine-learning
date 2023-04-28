import os
import numpy as np
import tensorflow as tf


fashion_mnist = tf.keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
train_images = train_images[..., None]
test_images = test_images[..., None]
train_images = train_images / np.float32(255)
test_images = test_images / np.float32(255)

strategy = tf.distribute.MirroredStrategy()
print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

BUFFER_SIZE = len(train_images)
BATCH_SIZE_PER_REPLICA = 64
GLOBAL_BATCH_SIZE = BATCH_SIZE_PER_REPLICA * strategy.num_replicas_in_sync

train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels))\
    .shuffle(BUFFER_SIZE)\
    .batch(GLOBAL_BATCH_SIZE)
test_dataset = tf.data.Dataset.from_tensor_slices((test_images, test_labels))\
    .batch(GLOBAL_BATCH_SIZE)
train_dist_dataset = strategy.experimental_distribute_dataset(train_dataset)
test_dist_dataset = strategy.experimental_distribute_dataset(test_dataset)


def step_train(inputs):
    with tf.GradientTape() as tape:
        images, labels = inputs
        predictions = model(images, training=True)
        per_example_loss = loss_object(labels, predictions)
        loss = tf.nn.compute_average_loss(per_example_loss, global_batch_size=GLOBAL_BATCH_SIZE)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        train_accuracy.update_state(labels, predictions)
        return loss


def step_test(inputs):
    images, labels = inputs
    predictions = model(images, training=False)
    loss = loss_object(labels, predictions)
    test_loss.update_state(loss)
    test_accuracy.update_state(labels, predictions)


@tf.function
def distributed_train_step(dataset_inputs):
    per_replica_losses = strategy.run(step_train, args=(dataset_inputs,))
    return strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)


@tf.function
def distributed_test_step(dataset_inputs):
    return strategy.run(step_test, args=(dataset_inputs,))


with strategy.scope():
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, 3, activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(64, 3, activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(10)
    ])

    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True,
        reduction=tf.keras.losses.Reduction.NONE
    )
    optimizer = tf.keras.optimizers.Adam()
    test_loss = tf.keras.metrics.Mean(name='test_loss')
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
    test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')
    checkpoint_dir = './training_checkpoints'
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")
    checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)

    EPOCHS = 12
    for epoch in range(EPOCHS):
        train_accuracy.reset_states()
        test_loss.reset_states()
        test_accuracy.reset_states()

        total_loss = 0.0
        num_batches = 0
        for x in train_dist_dataset:
            total_loss += distributed_train_step(x)
            num_batches += 1
        train_loss = total_loss / num_batches

        for x in test_dist_dataset:
            distributed_test_step(x)

        if epoch % 2 == 0:
            checkpoint.save(checkpoint_prefix)

        print(
            f'Epoch: {epoch + 1}, '
            f'Loss: {train_loss}, '
            f'Accuracy: {train_accuracy.result() * 100}, '
            f'Test Loss: {test_loss.result()}, '
            f'Test Accuracy: {test_accuracy.result() * 100}'
        )

