import tensorflow as tf


def _parse_record(example_proto):
    features = {
        "raw": tf.FixedLenFeature([], tf.string),
        "label": tf.FixedLenFeature([], tf.int64),
    }
    parsed_feature = tf.parse_single_example(example_proto, features=features)
    return parsed_feature


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
    dataset = tf.data.TFRecordDataset(["tmp1.tfrecord", "tmp2.tfrecord"])
    dataset = dataset.map(_parse_record)
    iterator = dataset.make_one_shot_iterator()

    for _ in range(4):
        features = sess.run(iterator.get_next())
        print(features["raw"], features["label"])
