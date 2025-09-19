from ultralytics import YOLO
import cv2


def video_detection(path, skip_frames=1):
    cap = cv2.VideoCapture(path)
    model_path = "../YOLOv8/runs/detect/train/weights/best.pt"
    model = YOLO(model_path)

    threshold = 0.5

    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        if frame_count % skip_frames != 0:
            continue

        results = model(frame)[0]

        for result in results.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = result

            if score > threshold:
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 200, 150), 4)
                cv2.putText(frame, results.names[int(class_id)].upper(), (int(x1), int(y1 - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 105), 1, cv2.LINE_AA)

        yield frame

    cap.release()
    cv2.destroyAllWindows()

