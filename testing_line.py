import cv2
import os
import pandas as pd
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

cap = cv2.VideoCapture('videos/cars2.mp4')

with open("classes.txt", "r") as class_file:
    data = class_file.read()
    class_list = data.split("\n")

frame_count = 0

tracker_car = Tracker()
tracker_bus = Tracker()
tracker_truck = Tracker()
line1_y = 250
line2_y = 270
offset = 8

car = {}
bus = {}
truck = {}

car_counter = []
bus_counter = []
truck_counter = []


while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_count += 1
    if frame_count % 3 != 0:
        continue
    frame = cv2.resize(frame, (1020, 500))
    results = model.predict(frame)
    boxes_data = results[0].boxes.data
    df = pd.DataFrame(boxes_data).astype("float")

    car_list = []
    bus_list = []
    truck_list = []

    for index, row in df.iterrows():
        x1 = int(row[0])
        y1 = int(row[1])
        x2 = int(row[2])
        y2 = int(row[3])
        class_idx = int(row[5])
        class_name = class_list[class_idx]
        if 'car' in class_name:
            car_list.append([x1, y1, x2, y2])

        elif 'bus' in class_name:
            bus_list.append([x1, y1, x2, y2])

        elif 'truck' in class_name:
            truck_list.append([x1, y1, x2, y2])

    tracked_car_bboxes = tracker_car.update(car_list)
    tracked_bus_bboxes = tracker_bus.update(bus_list)
    tracked_truck_bboxes = tracker_truck.update(truck_list)

    for tracked_bbox in tracked_car_bboxes:
        x3, y3, x4, y4, bbox_id = tracked_bbox
        center_x = int(x3 + x4) // 2
        center_y = int(y3 + y4) // 2
        if (center_y + offset) > line1_y > (center_y - offset):
            car[bbox_id] = (center_x, center_y)
        if bbox_id in car:
            if (center_y + offset) > line2_y > (center_y - offset):
                cv2.circle(frame, (center_x, center_y), 4, (255, 0, 0), -1)
                if car_counter.count(bbox_id) == 0:
                    car_counter.append(bbox_id)
        cv2.putText(frame, str(bbox_id), (x3, y3), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 0), 1)
        cv2.rectangle(frame, (x3, y3), (x4, y4), (255, 0, 255), 2)

    for tracked_bbox in tracked_bus_bboxes:
        x3, y3, x4, y4, bbox_id = tracked_bbox
        center_x = int(x3 + x4) // 2
        center_y = int(y3 + y4) // 2
        if (center_y + offset) > line1_y > (center_y - offset):
            car[bbox_id] = (center_x, center_y)
        if bbox_id in car:
            if (center_y + offset) > line2_y > (center_y - offset):
                cv2.circle(frame, (center_x, center_y), 4, (255, 0, 0), -1)
                cv2.rectangle(frame, (x3, y3), (x4, y4), (255, 0, 255), 2)
                if bus_counter.count(bbox_id) == 0:
                    bus_counter.append(bbox_id)
        cv2.putText(frame, str(bbox_id), (x3, y3), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 0), 1)
        cv2.rectangle(frame, (x3, y3), (x4, y4), (255, 0, 255), 2)

    for tracked_bbox in tracked_truck_bboxes:
        x3, y3, x4, y4, bbox_id = tracked_bbox
        center_x = int(x3 + x4) // 2
        center_y = int(y3 + y4) // 2
        if (center_y + offset) > line1_y > (center_y - offset):
            car[bbox_id] = (center_x, center_y)
        if bbox_id in car:
            if (center_y + offset) > line2_y > (center_y - offset):
                cv2.circle(frame, (center_x, center_y), 4, (255, 0, 0), -1)
                if truck_counter.count(bbox_id) == 0:
                    truck_counter.append(bbox_id)
        cv2.putText(frame, str(bbox_id), (x3, y3), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 0), 1)
        cv2.rectangle(frame, (x3, y3), (x4, y4), (255, 0, 255), 2)

    cv2.line(frame, (1, line1_y), (1018, line1_y), (0, 255, 0), 2)
    cv2.line(frame, (1, line2_y), (1018, line2_y), (0, 0, 255), 2)

    total_cars = len(car_counter)
    total_bus = len(bus_counter)
    total_trucks = len(truck_counter)

    cv2.putText(frame, f"Cars: {total_cars}", (50, 60), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(frame, f"Bus: {total_bus}", (50, 80), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(frame, f"Trucks: {total_trucks}", (50, 100), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

    cv2.imshow("Vehicle Tracking and Counting", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
