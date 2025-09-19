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

cap = cv2.VideoCapture('videos/sample_video.mp4')

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

car_up = {}
car_down = {}
bus_up = {}
bus_down = {}
truck_up = {}
truck_down = {}

car_counter_up = []
car_counter_down = []
bus_counter_up = []
bus_counter_down = []
truck_counter_up = []
truck_counter_down = []


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
            car_up[bbox_id] = (center_x, center_y)
        if bbox_id in car_up:
            if (center_y + offset) > line2_y > (center_y - offset):
                cv2.circle(frame, (center_x, center_y), 4, (255, 0, 0), -1)
                cv2.putText(frame, str(bbox_id), (x3, y3), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 0), 1)
                cv2.rectangle(frame, (x3, y3), (x4, y4), (255, 0, 255), 2)
                if car_counter_up.count(bbox_id) == 0:
                    car_counter_up.append(bbox_id)

        if (center_y + offset) > line2_y > (center_y - offset):
            car_down[bbox_id] = (center_x, center_y)
        if bbox_id in car_down:
            if (center_y + offset) > line1_y > (center_y - offset):
                cv2.circle(frame, (center_x, center_y), 4, (255, 0, 0), -1)
                cv2.putText(frame, str(bbox_id), (x3, y3), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 0), 1)
                cv2.rectangle(frame, (x3, y3), (x4, y4), (255, 0, 255), 2)
                if car_counter_down.count(bbox_id) == 0:
                    car_counter_down.append(bbox_id)

    for tracked_bbox in tracked_bus_bboxes:
        x3, y3, x4, y4, bbox_id = tracked_bbox
        center_x = int(x3 + x4) // 2
        center_y = int(y3 + y4) // 2
        if (center_y + offset) > line1_y > (center_y - offset):
            bus_up[bbox_id] = (center_x, center_y)
        if bbox_id in bus_up:
            if (center_y + offset) > line2_y > (center_y - offset):
                cv2.circle(frame, (center_x, center_y), 4, (255, 0, 0), -1)
                cv2.putText(frame, str(bbox_id), (x3, y3), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 0), 1)
                cv2.rectangle(frame, (x3, y3), (x4, y4), (255, 0, 255), 2)
                if bus_counter_up.count(bbox_id) == 0:
                    bus_counter_up.append(bbox_id)

        if (center_y + offset) > line2_y > (center_y - offset):
            bus_down[bbox_id] = (center_x, center_y)
        if bbox_id in bus_down:
            if (center_y + offset) > line1_y > (center_y - offset):
                cv2.circle(frame, (center_x, center_y), 4, (255, 0, 0), -1)
                cv2.putText(frame, str(bbox_id), (x3, y3), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 0), 1)
                cv2.rectangle(frame, (x3, y3), (x4, y4), (255, 0, 255), 2)
                if bus_counter_down.count(bbox_id) == 0:
                    bus_counter_down.append(bbox_id)

    for tracked_bbox in tracked_truck_bboxes:
        x3, y3, x4, y4, bbox_id = tracked_bbox
        center_x = int(x3 + x4) // 2
        center_y = int(y3 + y4) // 2
        if (center_y + offset) > line1_y > (center_y - offset):
            truck_up[bbox_id] = (center_x, center_y)
        if bbox_id in truck_up:
            if (center_y + offset) > line2_y > (center_y - offset):
                cv2.circle(frame, (center_x, center_y), 4, (255, 0, 0), -1)
                cv2.putText(frame, str(bbox_id), (x3, y3), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 0), 1)
                cv2.rectangle(frame, (x3, y3), (x4, y4), (255, 0, 255), 2)
                if truck_counter_up.count(bbox_id) == 0:
                    truck_counter_up.append(bbox_id)

        if (center_y + offset) > line2_y > (center_y - offset):
            truck_down[bbox_id] = (center_x, center_y)
        if bbox_id in truck_down:
            if (center_y + offset) > line1_y > (center_y - offset):
                cv2.circle(frame, (center_x, center_y), 4, (255, 0, 0), -1)
                cv2.putText(frame, str(bbox_id), (x3, y3), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 0), 1)
                cv2.rectangle(frame, (x3, y3), (x4, y4), (255, 0, 255), 2)
                if truck_counter_down.count(bbox_id) == 0:
                    truck_counter_down.append(bbox_id)

    cv2.line(frame, (1, line1_y), (1018, line1_y), (0, 255, 0), 2)
    cv2.line(frame, (1, line2_y), (1018, line2_y), (0, 0, 255), 2)

    total_cars_up = len(car_counter_up)
    total_cars_down = len(car_counter_down)
    total_bus_up = len(bus_counter_up)
    total_bus_down = len(bus_counter_down)
    total_trucks_up = len(truck_counter_up)
    total_trucks_down = len(truck_counter_down)

    cv2.putText(frame, f"Cars: {total_cars_up}", (50, 60), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(frame, f"Cars down: {total_cars_down}", (750, 60), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(frame, f"Bus: {total_bus_up}", (50, 90), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(frame, f"Bus: {total_bus_down}", (750, 90), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(frame, f"Trucks: {total_trucks_up}", (50, 120), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(frame, f"Trucks: {total_trucks_down}", (750, 120), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

    cv2.imshow("Vehicle Tracking and Counting", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
