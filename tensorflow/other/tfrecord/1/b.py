import tensorflow as tf

with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)

    # write
    with tf.python_io.TFRecordWriter("tmp.tfrecord") as writer:
        example = tf.train.Example(features=tf.train.Features(feature={
            "raw": tf.train.Feature(bytes_list=tf.train.BytesList(value=["Hello World!".encode("utf-8")])),
            "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[1])),
        }))
        writer.write(example.SerializeToString())

    # read
    queue = tf.train.string_input_producer(["tmp.tfrecord"], num_epochs=None)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(queue)
    features = tf.parse_single_example(serialized_example,  features={
        "raw": tf.FixedLenFeature([], tf.string),
        "label": tf.FixedLenFeature([], tf.int64),
    })

    raw = features["raw"]
    label = features["label"]
    raw, label = sess.run([raw, label])
    print(raw, label)

    coord.request_stop()
    coord.join(threads)
