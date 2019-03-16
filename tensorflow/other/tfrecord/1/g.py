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

    reader1 = tf.TFRecordReader()
    _, serialized_example1 = reader1.read(queue)
    features1 = tf.parse_single_example(serialized_example1, features={
        "raw": tf.FixedLenFeature([], tf.string),
        "label": tf.FixedLenFeature([], tf.int64),
    })

    reader2 = tf.TFRecordReader()
    _, serialized_example2 = reader2.read(queue)
    features2 = tf.parse_single_example(serialized_example2, features={
        "raw": tf.FixedLenFeature([], tf.string),
        "label": tf.FixedLenFeature([], tf.int64),
    })

    reader3 = tf.TFRecordReader()
    _, serialized_example3 = reader3.read(queue)
    features3 = tf.parse_single_example(serialized_example3, features={
        "raw": tf.FixedLenFeature([], tf.string),
        "label": tf.FixedLenFeature([], tf.int64),
    })

    for _ in range(2):
        raw = features1["raw"]
        label = features1["label"]
        raw, label = sess.run([raw, label])
        print(raw, label)

        raw = features2["raw"]
        label = features2["label"]
        raw, label = sess.run([raw, label])
        print(raw, label)

        raw = features3["raw"]
        label = features3["label"]
        raw, label = sess.run([raw, label])
        print(raw, label)
