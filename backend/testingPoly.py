import os
import cv2
import pandas as pd
import numpy as np
from ultralytics import YOLO
from tracker import *

model_path = os.path.join('.', 'runs', 'detect', 'train', 'weights', 'best.pt')
model = YOLO(model_path)


def mouse_position(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        point = [x, y]
        print(point)


cv2.namedWindow('Vehicle Tracking and Counting')
cv2.setMouseCallback('Vehicle Tracking and Counting', mouse_position)

cap = cv2.VideoCapture('videos/sample_video.mp4')

my_file = open("classes.txt", "r")
data = my_file.read()
class_list = data.split("\n")
# print(class_list)

count = 0

# area = [(295, 268), (476, 280), (468, 300), (278, 280)]
area = [(295, 268), (476, 280), (414, 493), (1, 434)]
tracker = Tracker()
area_c = set()

while True:
    ret, frame = cap.read()
    if not ret:
        break
    count += 1
    if count % 3 != 0:
        continue

    frame = cv2.resize(frame, (1020, 500))
    results = model.predict(frame)

    a = results[0].boxes.data
    px = pd.DataFrame(a).astype("float")

    car_list = []
    bus_list = []
    truck_list = []

    for index, row in px.iterrows():
        x1 = int(row[0])
        y1 = int(row[1])
        x2 = int(row[2])
        y2 = int(row[3])
        d = int(row[5])
        c = class_list[d]

        if 'car' in c:
            car_list.append([x1, y1, x2, y2])
        elif 'bus' in c:
            bus_list.append([x1, y1, x2, y2])
        elif 'truck' in c:
            truck_list.append([x1, y1, x2, y2])

    bbox_idx = tracker.update(car_list)

    for bbox in bbox_idx:
        x3, y3, x4, y4, bbox_id = bbox
        cx = int(x3 + x4) // 2
        cy = int(y3 + y4) // 2

        result = cv2.pointPolygonTest(np.array(area, np.int32), (cx, cy), False)
        if result >= 0:
            cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
            area_c.add(bbox_id)

        cv2.rectangle(frame, (x3, y3), (x4, y4), (0, 0, 255), 2)
        cv2.putText(frame, str(bbox_id), (x3, y3), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 0), 1)

    cv2.polylines(frame, [np.array(area, np.int32)], True, (80, 127, 255), 3)
    count = len(area_c)
    # print(len(area_c))
    cv2.putText(frame, str(count), (50, 80), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 255, 255), 2)
    cv2.imshow("Vehicle Tracking and Counting", frame)
    if cv2.waitKey(0) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
