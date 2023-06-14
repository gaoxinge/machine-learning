import cv2

net = cv2.dnn.readNetFromDarknet("yolov3.cfg", "yolov3.weights")
model = cv2.dnn_DetectionModel(net)
model.setInputParams(scale=1/255, size=(416, 416), swapRB=True)

img = cv2.imread("dog.jpg")
class_ids, scores, boxes = model.detect(img, 0.5, 0.4)
for (class_id, score, box) in zip(class_ids, scores, boxes):
    cv2.rectangle(img, box, (0, 0, 0), 1)
    cv2.putText(img, str(class_id), (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

cv2.imshow("", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

