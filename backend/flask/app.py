from flask import Flask, Response, jsonify, request
import cv2
from predict import video_detection

app = Flask(__name__)

def generate_frames(path = ''):
    yolo_output = video_detection(path)

    for det in yolo_output:
        ref, buffer = cv2.imencode('.jpg', det)
        frame = buffer.tobytes()

        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame +b'\r\n')


@app.route('/video')
def video():
    return Response(generate_frames(path='../YOLOv8/videos/sample_video.mp4'), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/webcam')
def webcam():
    return Response(generate_frames(path=0), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == "__main__":
    app.run(debug=True)
