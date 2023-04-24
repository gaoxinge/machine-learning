import tensorflow as tf
tf.debugging.set_log_device_placement(True)
print("CPU physical:", tf.config.experimental.list_physical_devices('CPU'))
print("CPU logical:", tf.config.list_logical_devices('CPU'))
print("GPU physical:", tf.config.experimental.list_physical_devices('GPU'))
print("GPU logical:", tf.config.list_logical_devices('GPU'))

print("=" * 70)
with tf.device('/CPU:0'):
    w = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    b = tf.constant([[1.0, 1.0], [1.0, 1.0]])
    s = tf.constant([[0.0, 0.0], [0.0, 0.0]])

print("=" * 70)
with tf.device('/GPU:0'):
    x = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])

print("=" * 70)
with tf.device('/GPU:0'):
    t = tf.matmul(w, x) + b

print("=" * 70)
with tf.device('/CPU:0'):
    s += t

print("=" * 70)
print(s)

