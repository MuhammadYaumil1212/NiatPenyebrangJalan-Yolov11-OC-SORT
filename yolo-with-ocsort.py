import cv2
from ultralytics import YOLO
from ocsort_tracker.ocsort import OCSort
import cvzone
import math
import supervision as sv

# initialize model and video
model = YOLO("../yolo-weights/yolo11l.pt")
cap = cv2.VideoCapture("videos/vid1.mp4")

# initialize trackers
# track = OCSort()
classNames = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
    "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop",
    "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
]

while True:
    success,img = cap.read()
    results = model(img,stream=True)
    for r in results:
        boxes = r.boxes
        for box in boxes:
            # bounding box
            x1,y1,x2,y2 = box.xyxy[0]
            x1,y1,x2,y2 = int(x1),int(y1),int(x2),int(y2)
            w, h = x2-x1, y2-y1
            cvzone.cornerRect(img,bbox=(x1,y1,w,h))
            # for confidence level
            conf = math.ceil((box.conf[0]*100))/100
            # class name
            cls = int(box.cls[0])
            cvzone.putTextRect(img,
                               f'{classNames[cls]} {conf}%',
                               (max(0, x1), max(35, y1)),
                               thickness=1,
                               scale=2
                               )
    cv2.imshow('img',img)
    cv2.waitKey(1)
