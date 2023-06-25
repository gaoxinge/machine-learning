import os
import cv2
import math
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import InputLayer, Conv2D, MaxPooling2D, BatchNormalization, LeakyReLU, Flatten, Dense
from tensorflow.keras.activations import sigmoid


def cal_iou(bboxes1, bboxes2):
    #bboxes: [batchsize, 7, 7, 2, 4] with [center_x, center_y, h, w]
    cx, cx_ = bboxes1[:, :, :, :, 0], bboxes2[:, :, :, :, 0]
    cy, cy_ = bboxes1[:, :, :, :, 1], bboxes2[:, :, :, :, 1]
    h, h_ = bboxes1[:, :, :, :, 2], bboxes2[:, :, :, :, 2]
    w, w_ = bboxes1[:, :, :, :, 3], bboxes2[:, :, :, :, 3]
    x1, x1_ = cx - w / 2, cx_ - w_ / 2
    x2, x2_ = cx + w / 2, cx_ + w_ / 2
    y1, y1_ = cy - h / 2, cy_ - h_ / 2
    y2, y2_ = cy + h / 2, cy_ + h_ / 2
    x_inter1 = tf.maximum(x1, x1_)
    x_inter2 = tf.minimum(x2, x2_)
    y_inter1 = tf.maximum(y1, y1_)
    y_inter2 = tf.minimum(y2, y2_)
    h_inter = tf.maximum(0., y_inter2 - y_inter1)
    w_inter = tf.maximum(0., x_inter2 - x_inter1)
    area_inter = h_inter * w_inter
    area_union = h * w + h_ * w_ - area_inter
    iou = area_inter / (area_union + 1e-6)
    return iou


def yolo_loss(prediction, labels, H=448, W=448, S=7):
    #prediction: batchsize x 7 x 7 x 30, bboxes: batchsize x 7 x 7 x (0:8), confidences: batchsize x 7 x 7 x (8:10), class: batchsize x 7 x 7 x (10:30)
    #labels: batchsize x 7 x 7 x 25, response_mask: batchsize x 7 x 7 x (0:1), bbox: batchsize x 7 x 7 x (1:5), class: batchsize: batchsize x 7 x 7 x (5:25)
    prediction = tf.reshape(prediction, [-1, 7, 7, 30])
    pred_bboxes = prediction[:, :, :, 0:8]
    pred_bboxes = tf.reshape(pred_bboxes, [-1, 7, 7, 2, 4])
    pred_confidence = prediction[:, :, :, 8:10]
    pred_class = prediction[:, :, :, 10:]
    #groundtruth
    label_mask = labels[:, :, :, 4:5]
    label_bboxes = labels[:, :, :, 0:4]
    label_bboxes = tf.tile(label_bboxes, multiples=[1, 1, 1, 2])
    label_bboxes = tf.reshape(label_bboxes, [-1, 7, 7, 2, 4])
    label_class = labels[:, :, :, 5:]
    #prediction normalized offset -> [c_x, c_y, h, w]
    cell_h = H / S
    cell_w = W / S
    temp = tf.constant([[0., 1., 2., 3., 4., 5., 6.]])
    temp = tf.tile(temp, multiples=[7, 1])
    col = temp[tf.newaxis, :, :, tf.newaxis, tf.newaxis]#dimension 7 x 7 -> 1 x 7 x 7 x 1 x 1
    row = tf.transpose(col, perm=[0, 2, 1, 3, 4])
    pred_bboxes_original = tf.concat([(pred_bboxes[:, :, :, :, 0:1] + col) * cell_w,
                                      (pred_bboxes[:, :, :, :, 1:2] + row) * cell_h,
                                       tf.square(pred_bboxes[:, :, :, :, 2:3]) * H,
                                       tf.square(pred_bboxes[:, :, :, :, 3:4]) * W], axis=-1)
    label_bboxes_original = tf.concat([(label_bboxes[:, :, :, :, 0:1] + col) * cell_w,
                                      (label_bboxes[:, :, :, :, 1:2] + row) * cell_h,
                                      tf.square(label_bboxes[:, :, :, :, 2:3]) * H,
                                      tf.square(label_bboxes[:, :, :, :, 3:4]) * W], axis=-1)
    iou = cal_iou(pred_bboxes_original, label_bboxes_original)#input: batchsize x 7 x 7 x 2 x 4, output: batchsize x 7 x 7 x 2
    max_iou = tf.reduce_max(iou, axis=-1, keepdims=True)#output: batchsize x 7 x 7 x 1
    mask_obj = tf.cast(tf.greater_equal(iou, max_iou), dtype=tf.float32) * label_mask
    loss_bboxes = tf.reduce_mean(tf.reduce_sum(tf.square(pred_bboxes - label_bboxes) * mask_obj[:, :, :, :, tf.newaxis], axis=[1, 2, 3, 4]))
    loss_confidence_obj = tf.reduce_mean(tf.reduce_sum(tf.square(pred_confidence - 1) * mask_obj, axis=[1, 2, 3]))
    loss_confidence_noobj = tf.reduce_mean(tf.reduce_sum(tf.square(pred_confidence) * (1 - mask_obj), axis=[1, 2, 3]))
    loss_class = tf.reduce_mean(tf.reduce_sum(tf.square(pred_class - label_class) * label_mask, axis=[1, 2, 3]))
    return loss_bboxes, loss_confidence_obj, loss_confidence_noobj, loss_class


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
        # self.nn.append(sigmoid)
    
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

        result = np.zeros((self.S, self.S, 5 + self.C), dtype=np.float32)
        for label in labels:
            c, cx, cy, dw, dh = label
            cx *= self.S
            cy *= self.S
            col = int(cx)
            row = int(cy)
            dw = math.sqrt(dw)
            dh = math.sqrt(dh)
            result[row, col] = [cx - col, cy - row, dw, dh, 1] + [(1 if _ == c else 0) for _ in range(self.C)]
        result = tf.constant(result)

        return img, result

    def detect(self, preds):
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
        return pred_class, pred_conf, pred_box1

    def loss(self, preds, labels):
        return yolo_loss(preds, labels)


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
    preds[:, :, 8:9] = labels[:, :, 4:5]
    preds[:, :, 9:10] = labels[:, :, 4:5]
    preds[:, :, 10:30] = labels[:, :, 5:]
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

    loss1, loss2, loss3, loss4 =  yolo_v1.loss(preds, labels)
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

