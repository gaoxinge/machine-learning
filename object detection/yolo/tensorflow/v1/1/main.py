import os
import tqdm
import cv2
import numpy as np
import tensorflow as tf
from model import YoloV1


class PascalVoc(tf.data.Dataset):

    def _generator(paths):
        for path in paths:
            path = path.decode("utf-8")
            a, b = os.path.split(path)
            c, d = os.path.split(a)
            e, f = os.path.splitext(b)
            image_path = os.path.join(c, "JPEGImages", e + ".jpg")
            label_path = os.path.join(c, "labels", e + ".txt")
            img = cv2.imread(image_path)
            labels = []
            with open(label_path, "r", encoding="utf-8") as f:
                for line in f:
                    c, cx, cy, dw, dh = line.strip().split(" ")
                    c, cx, cy, dw, dh = int(c), float(cx), float(cy), float(dw), float(dh)
                    labels.append([c, cx, cy, dw, dh])
            img, labels = yolo_v1.preprocess(img, labels)
            yield img, labels

    def __new__(cls, filepath):
        with open(filepath, "r", encoding="utf-8") as f:
            paths = [line.strip() for line in f]
        return tf.data.Dataset.from_generator(
            cls._generator,
            output_shapes=(tf.TensorShape((448, 448, 3)), tf.TensorShape((7, 7, 24))), 
            output_types=(tf.float32, tf.float32),
            args=(paths,)
        )


yolo_v1 = YoloV1()
optimizer = tf.keras.optimizers.SGD(learning_rate=0.0001, momentum=0.9, weight_decay=0.0005)
checkpoint_dir = "./training_checkpoints"
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(model=yolo_v1)


@tf.function
def step_train(img, labels):
    with tf.GradientTape() as tape:
        preds = yolo_v1(img, training=True)
        loss1, loss2, loss3, loss4 = yolo_v1.loss(preds, labels)
        loss = 5 * loss1 + loss2 + 0.5 * loss3 + loss4
        gradients = tape.gradient(loss, yolo_v1.trainable_variables)
        optimizer.apply_gradients(zip(gradients, yolo_v1.trainable_variables))
        return loss, loss1, loss2, loss3, loss4


train_d = PascalVoc("data/train.txt").shuffle(128).batch(16)
cnt = 0
loss = 0
loss1 = 0
loss2 = 0
loss3 = 0
loss4 = 0
for epoch in range(20):
    for img, labels in train_d:
        loss_, loss1_, loss2_, loss3_, loss4_ = step_train(img, labels)
        cnt += 1
        loss = loss * 0.9 + loss_.numpy() * 0.1
        loss1 = loss1 * 0.9 + loss1_.numpy() * 0.1
        loss2 = loss2 * 0.9 + loss2_.numpy() * 0.1
        loss3 = loss3 * 0.9 + loss3_.numpy() * 0.1
        loss4 = loss4 * 0.9 + loss4_.numpy() * 0.1
        if cnt % 500 == 0:
            checkpoint.save(checkpoint_prefix)
        print(cnt, loss, loss1, loss2, loss3, loss4)

