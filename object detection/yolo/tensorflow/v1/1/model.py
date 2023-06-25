import os
import cv2
import math
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import InputLayer, Conv2D, MaxPooling2D, BatchNormalization, LeakyReLU, Flatten, Dense
from tensorflow.keras.activations import sigmoid


def get_iou(box, box_):
    """
    box1: ? * S * S * B * 4
    box2: ? * S * S * B * 4
    return: ? * S * S * B
    """
    xy1 = box[:, :, :, :, 0:2] - box[:, :, :, :, 2:4] / 2
    xy2 = box[:, :, :, :, 0:2] + box[:, :, :, :, 2:4] / 2
    xy_1 = box_[:, :, :, :, 0:2] - box[:, :, :, :, 2:4] / 2
    xy_2 = box_[:, :, :, :, 0:2] + box[:, :, :, :, 2:4] / 2
    
    ixy1 = tf.maximum(xy1, xy_1)
    ixy2 = tf.minimum(xy2, xy_2)
    iwh = tf.maximum(0., ixy2 - ixy1)
    
    i = iwh[:, :, :, :, 0] * iwh[:, :, :, :, 1]
    u = box[:, :, :, :, 2] * box[:, :, :, :, 3] + box_[:, :, :, :, 2] * box_[:, :, :, :, 3] - i

    return i / (u + 1e-6)


class YoloV1(Model):

    def __init__(self, W=448, H=448, S=7, B=2, C=20):
        super(YoloV1, self).__init__()
        self.W = W
        self.H = H
        self.S = S
        self.B = B
        self.C = C

        col = tf.reshape(tf.tile(tf.range(0, self.S, dtype=tf.float32), [self.S]), (self.S, self.S))
        row = tf.transpose(col)
        self.col = col[tf.newaxis, :, :, tf.newaxis, tf.newaxis]
        self.row = row[tf.newaxis, :, :, tf.newaxis, tf.newaxis]

        self.nn = []
        self.nn.append(InputLayer(input_shape=(self.W, self.H, 3)))
        self.nn.append(Conv2D(filters=64, kernel_size=(7, 7), strides=(2, 2), padding="same"))
        self.nn.append(LeakyReLU(alpha=0.1))
        self.nn.append(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="same"))
        self.nn.append(Conv2D(filters=192, kernel_size=(3, 3), strides=(1, 1), padding="same"))
        self.nn.append(LeakyReLU(alpha=0.1))
        self.nn.append(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="same"))
        self.nn.append(Conv2D(filters=128, kernel_size=(1, 1), strides=(1, 1), padding="same"))
        self.nn.append(LeakyReLU(alpha=0.1))
        self.nn.append(Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding="same"))
        self.nn.append(LeakyReLU(alpha=0.1))
        self.nn.append(Conv2D(filters=256, kernel_size=(1, 1), strides=(1, 1), padding="same"))
        self.nn.append(LeakyReLU(alpha=0.1))
        self.nn.append(Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), padding="same"))
        self.nn.append(LeakyReLU(alpha=0.1))
        self.nn.append(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="same"))
        self.nn.append(Conv2D(filters=256, kernel_size=(1, 1), strides=(1, 1), padding="same"))
        self.nn.append(LeakyReLU(alpha=0.1))
        self.nn.append(Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), padding="same"))
        self.nn.append(LeakyReLU(alpha=0.1))
        self.nn.append(Conv2D(filters=256, kernel_size=(1, 1), strides=(1, 1), padding="same"))
        self.nn.append(LeakyReLU(alpha=0.1))
        self.nn.append(Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), padding="same"))
        self.nn.append(LeakyReLU(alpha=0.1))
        self.nn.append(Conv2D(filters=256, kernel_size=(1, 1), strides=(1, 1), padding="same"))
        self.nn.append(LeakyReLU(alpha=0.1))
        self.nn.append(Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), padding="same"))
        self.nn.append(LeakyReLU(alpha=0.1))
        self.nn.append(Conv2D(filters=256, kernel_size=(1, 1), strides=(1, 1), padding="same"))
        self.nn.append(LeakyReLU(alpha=0.1))
        self.nn.append(Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), padding="same"))
        self.nn.append(LeakyReLU(alpha=0.1))
        self.nn.append(Conv2D(filters=512, kernel_size=(1, 1), strides=(1, 1), padding="same"))
        self.nn.append(LeakyReLU(alpha=0.1))
        self.nn.append(Conv2D(filters=1024, kernel_size=(3, 3), strides=(1, 1), padding="same"))
        self.nn.append(LeakyReLU(alpha=0.1))
        self.nn.append(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="same"))
        self.nn.append(Conv2D(filters=512, kernel_size=(1, 1), strides=(1, 1), padding="same"))
        self.nn.append(LeakyReLU(alpha=0.1))
        self.nn.append(Conv2D(filters=1024, kernel_size=(3, 3), strides=(1, 1), padding="same"))
        self.nn.append(LeakyReLU(alpha=0.1))
        self.nn.append(Conv2D(filters=512, kernel_size=(1, 1), strides=(1, 1), padding="same"))
        self.nn.append(LeakyReLU(alpha=0.1))
        self.nn.append(Conv2D(filters=1024, kernel_size=(3, 3), strides=(1, 1), padding="same"))
        self.nn.append(LeakyReLU(alpha=0.1))
        self.nn.append(Conv2D(filters=1024, kernel_size=(3, 3), strides=(1, 1), padding="same"))
        self.nn.append(LeakyReLU(alpha=0.1))
        self.nn.append(Conv2D(filters=1024, kernel_size=(3, 3), strides=(2, 2), padding="same"))
        self.nn.append(LeakyReLU(alpha=0.1))
        self.nn.append(Conv2D(filters=1024, kernel_size=(3, 3), strides=(1, 1), padding="same"))
        self.nn.append(LeakyReLU(alpha=0.1))
        self.nn.append(Conv2D(filters=1024, kernel_size=(3, 3), strides=(1, 1), padding="same"))
        self.nn.append(LeakyReLU(alpha=0.1))
        self.nn.append(Flatten())
        self.nn.append(Dense(units=4096))
        self.nn.append(LeakyReLU(alpha=0.1))
        self.nn.append(Dense(units=self.S * self.S * (self.B * 5 + self.C)))
        self.nn.append(sigmoid)

    def call(self, x):
        for layer in self.nn:
            x = layer(x)
        return x
    
    def preprocess(self, img, labels=None):
        img = cv2.resize(img, (self.H, self.W))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = tf.cast(img, tf.float32) / 255

        if labels is None:
            return img, None

        result = np.zeros((self.S, self.S, 4 + self.C), dtype=np.float32)
        for label in labels:
            c, cx, cy, dw, dh = label
            cx *= self.S
            cy *= self.S
            col = int(cx)
            row = int(cy)
            dw = math.sqrt(dw)
            dh = math.sqrt(dh)
            result[row, col] = [cx - col, cy - row, dw, dh] + [(1 if _ == c else 0) for _ in range(self.C)]
        result = tf.constant(result)

        return img, result

    def _common_preds(self, preds):
        preds = tf.reshape(preds, [-1, self.S, self.S, self.B * 5 + self.C])  # ? * S * S * (B * 5 + C)
        pred_box = preds[:, :, :, :self.B * 4]                                # ? * S * S * (B * 4)
        pred_box = tf.reshape(pred_box, [-1, self.S, self.S, self.B, 4])      # ? * S * S * B * 4
        pred_conf = preds[:, :, :, self.B * 4:self.B * 5]                     # ? * S * S * B
        pred_class = preds[:, :, :, self.B * 5:]                              # ? * S * S * C
        pred_box1 = tf.concat([
            (pred_box[:, :, :, :, 0:1] + self.col) / self.S,
            (pred_box[:, :, :, :, 1:2] + self.row) / self.S,
            tf.square(pred_box[:, :, :, :, 2:3]),
            tf.square(pred_box[:, :, :, :, 3:4]),
        ], axis=-1)
        return pred_class, pred_conf, pred_box, pred_box1

    def _common_labels(self, labels):
        label_box = labels[:, :, :, :4]                                     # ? * S * S * 4
        label_box = tf.tile(label_box, [1, 1, 1, self.B])                   # ? * S * S * (B * 4)
        label_box = tf.reshape(label_box, [-1, self.S, self.S, self.B, 4])  # ? * S * S * B * 4
        label_class = labels[:, :, :, 4:]                                   # ? * S * S * C
        label_box1 = tf.concat([
            (label_box[:, :, :, :, 0:1] + self.col) / self.S,
            (label_box[:, :, :, :, 1:2] + self.row) / self.S,
            tf.square(label_box[:, :, :, :, 2:3]),
            tf.square(label_box[:, :, :, :, 3:4]),
        ], axis=-1)
        return label_class, label_box, label_box1

    def detect(self, preds):
        pred_class, pred_conf, _, pred_box1 = self._common_preds(preds)
        return pred_class, pred_conf, pred_box1

    def loss(self, preds, labels):
        """
        preds: ? * (S * S * (B * 5 + C)) / ? * S * S * (B * 5 + C)
        labels: ? * S * S * (4 + C)
        return: 1
        """
        pred_class, pred_conf, pred_box, pred_box1 = self._common_preds(preds)
        label_class, label_box, label_box1 = self._common_labels(labels)
        
        label_mask = tf.reduce_max(label_class, axis=-1, keepdims=True) > 0  # ? * S * S * 1
        iou = get_iou(pred_box1, label_box1)                                 # ? * S * S * B
        iou_max = tf.reduce_max(iou, axis=-1, keepdims=True)                 # ? * S * S * 1
        obj_mask = tf.logical_and(iou >= iou_max, label_mask)                # ? * S * S * B

        """
        loss_box = 2 * tf.nn.l2_loss((pred_box - label_box) * obj_mask[:, :, :, :, tf.newaxis])
        loss_conf1 = 2 * tf.nn.l2_loss((1 - pred_conf) * obj_mask)
        loss_conf2 = 2 * tf.nn.l2_loss(pred_conf * (1 - obj_mask))
        loss_class = 2 * tf.nn.l2_loss((pred_class - label_class) * label_mask)
        iou = tf.reduce_sum(iou * label_mask)
        """
        loss_box = tf.reduce_mean(tf.reduce_sum(tf.square(tf.where(obj_mask[:, :, :, :, tf.newaxis], pred_box - label_box,  0))))
        loss_conf1 = tf.reduce_mean(tf.reduce_sum(tf.square(tf.where(obj_mask, 1 - pred_conf, 0))))
        loss_conf2 = tf.reduce_mean(tf.reduce_sum(tf.square(tf.where(tf.logical_not(obj_mask), pred_conf, 0))))
        loss_class = tf.reduce_mean(tf.reduce_sum(tf.square(tf.where(label_mask, pred_class - label_class, 0))))
        return loss_box, loss_conf1, loss_conf2, loss_class


def _check_model1():
    yolo_v1 = YoloV1()

    path = "data/VOCdevkit/VOC2007/"
    image = "JPEGImages/009962.jpg"
    label = "labels/009962.txt"
    image_path = os.path.join(path, image)
    label_path = os.path.join(path, label)

    img = cv2.imread(image_path)
    h, w = img.shape[:2]
    labels = []
    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            c, cx, cy, dw, dh = line.strip().split(" ")
            c, cx, cy, dw, dh = int(c), float(cx), float(cy), float(dw), float(dh)
            labels.append((c, cx, cy, dw, dh))

    img, labels = yolo_v1.preprocess(img, labels)
    labels = labels.numpy()
    preds = np.zeros((7, 7, 30), dtype=np.float32)
    preds[:, :, :4] = labels[:, :, :4]
    preds[:, :, 4:8] = labels[:, :, :4]
    preds[:, :, 8:9] = 1
    preds[:, :, 9:10] = 1
    preds[:, :, 10:30] = labels[:, :, 4:]
    preds = tf.constant(preds)
    preds = preds[tf.newaxis, :, :, :]

    img = img.numpy()
    img *= 255
    img = img.astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img = cv2.resize(img, (w, h))
    pred_class, pred_conf, pred_box = yolo_v1.detect(preds)
    pred_class, pred_conf, pred_box = pred_class[0], pred_conf[0], pred_box[0]
    for col in range(7):
        for row in range(7):
            pb = pred_box[row, col]
            pb *= np.array([w, h, w, h])
            for b in pb:
                cx, cy, dw, dh = b
                cv2.rectangle(img, (int(cx - dw / 2), int(cy - dh / 2)), (int(cx + dw / 2), int(cy + dh / 2)), (0, 0, 0), 1)
    cv2.imwrite("test.jpg", img)


def _check_model2():
    yolo_v1 = YoloV1()

    img = np.zeros((400, 400, 3), dtype=np.float32)
    labels = [(0, 0.5, 0.5, 0.25, 0.25)]
    img, labels = yolo_v1.preprocess(img, labels)
    labels = labels[tf.newaxis, :, :, :]

    preds = np.zeros((7, 7, 30), dtype=np.float32)
    preds[3, 3] = [0.5, 0.5, math.sqrt(0.125), math.sqrt(0.25), 0.5, 0.5, math.sqrt(0.25), math.sqrt(0.125), 1, 1] + [(1 if _ == 0 else 0) for _ in range(20)]
    preds = preds[tf.newaxis, :, :, :]

    loss1, loss2, loss3, loss4 = yolo_v1.loss(preds, labels)
    print(loss1.numpy(), loss2.numpy(), loss3.numpy(), loss4.numpy())


def _check_model3():
    yolo_v1 = YoloV1()

    path = "data/VOCdevkit/VOC2007/"
    image = "JPEGImages/009962.jpg"
    label = "labels/009962.txt"
    image_path = os.path.join(path, image)
    label_path = os.path.join(path, label)

    img = cv2.imread(image_path)
    h, w = img.shape[:2]
    labels = []
    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            c, cx, cy, dw, dh = line.strip().split(" ")
            c, cx, cy, dw, dh = int(c), float(cx), float(cy), float(dw), float(dh)
            labels.append((c, cx, cy, dw, dh))
    img, labels = yolo_v1.preprocess(img, labels)
    labels = labels[tf.newaxis, :, :, :]

    preds = tf.zeros((1, 7, 7, 30), dtype=tf.float32)
    loss1, loss2, loss3, loss4 = yolo_v1.loss(preds, labels)
    print(loss1.numpy(), loss2.numpy(), loss3.numpy(), loss4.numpy())

    preds = yolo_v1(img[tf.newaxis, :, :, :])
    loss1, loss2, loss3, loss4 = yolo_v1.loss(preds, labels)
    print(loss1.numpy(), loss2.numpy(), loss3.numpy(), loss4.numpy())


if __name__ == "__main__":
    _check_model1()
    _check_model2()
    _check_model3()

