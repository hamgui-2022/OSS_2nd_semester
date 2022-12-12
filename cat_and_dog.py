import cv2
import numpy as np



# read image
img = cv2.imread('./image/cat.jpg')
height, width, channel = img.shape


# get blob from image
blob = cv2.dnn.blobFromImage(img, 1 / 255, (416, 416), (0, 0, 0), swapRB=True, crop=False)


# read coco object names
with open("coco.names.txt", "r") as f:
    classes = [line.strip() for line in f.readlines()]



# load pre-trained yolo model from configuration and weight files
net = cv2.dnn.readNetFromDarknet('yolov3.cfg.txt', 'yolov3.weights')
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)

# set output layers
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]


# detect objects
net.setInput(blob)
outs = net.forward(output_layers)

# get bounding boxes and confidence socres
class_ids = []
confidence_scores = []
boxes = []

for out in outs: # for each detected object

    for detection in out: # for each bounding box

        scores = detection[5:] # scores (confidence) for all classes
        class_id = np.argmax(scores) # class id with the maximum score (confidence)
        confidence = scores[class_id] # the maximum score

        if confidence > 0.5:
            # bounding box coordinates
            center_x = int(detection[0] * width)
            center_y = int(detection[1] * height)
            w = int(detection[2] * width)
            h = int(detection[3] * height)

            # rectangle coordinates
            x = int(center_x - w / 2)
            y = int(center_y - h / 2)

            boxes.append([x, y, w, h])
            confidence_scores.append(float(confidence))
            class_ids.append(class_id)



# non maximum suppression
indices = cv2.dnn.NMSBoxes(boxes, confidence_scores, 0.5, 0.4)


# draw bounding boxes with labels on image
colors = np.random.uniform(0, 255, size=(len(classes), 3))
font = cv2.FONT_HERSHEY_PLAIN


# read image
img2 = cv2.imread('./image/puppy.jpg')
height2, width2, channel = img2.shape


# get blob from image
blob = cv2.dnn.blobFromImage(img2, 1 / 255, (416, 416), (0, 0, 0), swapRB=True, crop=False)


# read coco object names
with open("coco.names.txt", "r") as f:
    classes = [line.strip() for line in f.readlines()]



# load pre-trained yolo model from configuration and weight files
net = cv2.dnn.readNetFromDarknet('yolov3.cfg.txt', 'yolov3.weights')
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)

# set output layers
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]


# detect objects
net.setInput(blob)
outs = net.forward(output_layers)

# get bounding boxes and confidence socres
class_ids = []
confidence_scores = []
boxes = []

for out in outs: # for each detected object

    for detection in out: # for each bounding box

        scores = detection[5:] # scores (confidence) for all classes
        class_id = np.argmax(scores) # class id with the maximum score (confidence)
        confidence = scores[class_id] # the maximum score

        if confidence > 0.5:
            # bounding box coordinates
            center_x = int(detection[0] * width)
            center_y = int(detection[1] * height)
            w2 = int(detection[2] * width)
            h2= int(detection[3] * height)

            # rectangle coordinates
            x2 = int(center_x - w / 2)
            y2 = int(center_y - h / 2)

            boxes.append([x, y, w, h])
            confidence_scores.append(float(confidence))
            class_ids.append(class_id)



# non maximum suppression
indices = cv2.dnn.NMSBoxes(boxes, confidence_scores, 0.5, 0.4)


# draw bounding boxes with labels on image
colors = np.random.uniform(0, 255, size=(len(classes), 3))
font = cv2.FONT_HERSHEY_PLAIN



img2 = img2[y2:y2+height2, x2:x2+width2].copy()
img = img[y:y+height, x:x+width].copy()


img = cv2.resize(img, dsize=(500,400) , interpolation=cv2.INTER_AREA)
img2 = cv2.resize(img2, dsize=(500,400) , interpolation=cv2.INTER_AREA)
result=cv2.add(img, img2)




cv2.imshow("Cats", img)
cv2.imshow("Dogs", img2)
cv2.imshow("Mix", result)
cv2.waitKey(0)
cv2.destroyAllWindows()

