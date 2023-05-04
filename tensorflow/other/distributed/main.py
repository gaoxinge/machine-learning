import tensorflow as tf


@tf.function
def gather_value(x):
    with tf.GradientTape() as tape:
        ctx = tf.distribute.get_replica_context()
        loss = tf.reduce_sum(x @ w)
        dl_dw = tape.gradient(loss, w)
        reduced = ctx.all_reduce(tf.distribute.ReduceOp.SUM, dl_dw)
        w.assign_add(lr * reduced)


strategy = tf.distribute.MirroredStrategy()
with strategy.scope():
    w = tf.Variable([[1.], [1.], [1.]], name='w')
    x = tf.data.Dataset.from_tensor_slices([[1., 2., 3.], [1., 2., 3.], [1., 2., 3.], [1., 2., 3.]]).batch(2)
    x = strategy.experimental_distribute_dataset(x)
    lr = 1

    for t in x:
        print(t)
        strategy.run(gather_value, args=(t,))
    print(w)
