import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))


@tf.function(jit_compile=True)
def model_fn(x, y, z):
    return x + y * z


if __name__ == "__main__":
    a = tf.ones([1, 10])
    r = model_fn(a, a, a)
    print(r)

