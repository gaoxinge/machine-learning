import tensorflow as tf

with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)

    # write
    with tf.python_io.TFRecordWriter("tmp1.tfrecord") as writer:
        for raw, label in [("Hello World!", 1), ("FUCK", 2)]:
            example = tf.train.Example(features=tf.train.Features(feature={
                "raw": tf.train.Feature(bytes_list=tf.train.BytesList(value=[raw.encode("utf-8")])),
                "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[label])),
            }))
            writer.write(example.SerializeToString())

    with tf.python_io.TFRecordWriter("tmp2.tfrecord") as writer:
        for raw, label in [("Hi", 3), ("Ha", 4)]:
            example = tf.train.Example(features=tf.train.Features(feature={
                "raw": tf.train.Feature(bytes_list=tf.train.BytesList(value=[raw.encode("utf-8")])),
                "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[label])),
            }))
            writer.write(example.SerializeToString())

    # read
    queue = tf.train.string_input_producer(["tmp1.tfrecord", "tmp2.tfrecord"], num_epochs=None)
    tf.train.start_queue_runners(sess=sess)

    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(queue)
    features = tf.parse_single_example(serialized_example, features={
        "raw": tf.FixedLenFeature([], tf.string),
        "label": tf.FixedLenFeature([], tf.int64),
    })

    for _ in range(4):
        raw = features["raw"]
        label = features["label"]
        raw, label = sess.run([raw, label])
        print(raw, label)
