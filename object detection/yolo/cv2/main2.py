import cv2
import numpy as np


def extract(outputs, w, h):
    class_ids = []
    confidences = []
    boxes = []

    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            class_ids.append(class_id)
            confidences.append(float(confidence))

            box = detection[:4] * np.array([w, h, w, h])
            (center_x, center_y, width, height) = box
            x = center_x - (width / 2)
            y = center_y - (height / 2)
            box = [int(x), int(y), int(width), int(height)]
            boxes.append(box)

    return class_ids, confidences, boxes


def extract_vector(outputs, w, h):
    class_ids = None
    confidences = None
    boxes = None

    for output in outputs:
        scores = output[:, 5:]
        class_id = np.argmax(scores, axis=-1)
        confidence = scores[np.arange(scores.shape[0]), class_id]
        if class_ids is None:
            class_ids = class_id
        else:
            class_ids = np.append(class_ids, class_id, axis=0)
        if confidences is None:
            confidences = confidence
        else:
            confidences = np.append(confidences, confidence)

        box = output[:, :4] * np.array([w, h, w, h])
        box[:, 0] -= box[:, 2] / 2
        box[:, 1] -= box[:, 3] / 2
        box = box.astype(np.int32)
        if boxes is None:
            boxes = box
        else:
            boxes = np.append(boxes, box, axis=0)

    return class_ids, confidences, boxes


def iou(box1, box2):
    (x1, y1, w1, h1) = box1
    (x2, y2, w2, h2) = box2

    a1 = w1 * h1
    a2 = w2 * h2

    xx1 = max(x1, x2)
    yy1 = max(y1, y2)
    xx2 = min(x1 + w1, x2 + w2)
    yy2 = min(y1 + h1, y2 + h2)

    w = max(0, xx2 - xx1)
    h = max(0, yy2 - yy1)

    i = w * h
    u = a1 + a2 - i
    return i / u


def nms(boxes, confidences, thresh1, thresh2):
    box_confs = []
    for i in range(len(boxes)):
        confidence = confidences[i]
        if confidence < thresh1:
            continue
        box_confs.append((i, confidence))

    box_confs = sorted(box_confs, key=lambda box_conf: box_conf[1], reverse=True)
    indices = []
    while len(box_confs) > 0:
        box_conf = box_confs.pop(0)
        indices.append(box_conf[0])
        rs = []
        for i, box_conf1 in enumerate(box_confs):
            r = iou(boxes[box_conf[0]], boxes[box_conf1[0]])
            if r < thresh2:
                continue
            rs.append(i)
        for i in reversed(rs):
            box_confs.pop(i)
    return indices


net = cv2.dnn.readNetFromDarknet("yolov3.cfg", "yolov3.weights")
ln = net.getLayerNames()
ln = [ln[i - 1] for i in net.getUnconnectedOutLayers()]

img = cv2.imread("dog.jpg")
blob = cv2.dnn.blobFromImage(img, 1 / 255, (416, 416), swapRB=True, crop=False)
net.setInput(blob)
outputs = net.forward(ln)

h, w = img.shape[:2]
# class_ids, confidences, boxes = extract(outputs, w, h)
class_ids, confidences, boxes = extract_vector(outputs, w, h)

# indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
indices = nms(boxes, confidences, 0.5, 0.4)
for i in indices:
    box = boxes[i]
    class_id = class_ids[i]
    (x, y, w, h) = box
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 0), 1)
    cv2.putText(img, str(class_id), (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)


cv2.imshow("", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

