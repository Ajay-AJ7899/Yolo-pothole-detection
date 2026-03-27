import cv2 as cv
import numpy as np
import base64
import os
from flask import Flask, render_template, jsonify, request

app = Flask(__name__)

# ─── Load the same YOLOv4-tiny model used in camera_video.py ────────────────
print("Loading YOLOv4-tiny model...")
net = cv.dnn.readNet('project_files/yolov4_tiny.weights', 'project_files/yolov4_tiny.cfg')
net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)

model = cv.dnn_DetectionModel(net)
model.setInputParams(size=(640, 480), scale=1/255, swapRB=True)
print("Model loaded!")

Conf_threshold = 0.5
NMS_threshold = 0.4


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/process_frame', methods=['POST'])
def process_frame():
    """Receive a webcam frame from the browser, detect potholes, return annotated frame."""
    try:
        img_data = request.json['image'].split(',')[1]
        img_bytes = base64.b64decode(img_data)
        frame = cv.imdecode(np.frombuffer(img_bytes, np.uint8), cv.IMREAD_COLOR)

        if frame is None:
            return jsonify({'error': 'Bad image'}), 400

        h, w = frame.shape[:2]
        detections = []

        classes, scores, boxes = model.detect(frame, Conf_threshold, NMS_threshold)
        for (classid, score, box) in zip(classes, scores, boxes):
            x, y, bw, bh = box
            if score >= 0.7 and (bw * bh) / (w * h) <= 0.1 and y < 600:
                cv.rectangle(frame, (x, y), (x + bw, y + bh), (0, 255, 0), 2)
                cv.putText(frame, f"{round(float(score)*100,1)}% pothole",
                           (x, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                detections.append({'confidence': float(score)})

        _, buf = cv.imencode('.jpg', frame)
        b64 = base64.b64encode(buf).decode()

        return jsonify({
            'image': f'data:image/jpeg;base64,{b64}',
            'detections': detections,
            'count': len(detections)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    print("\n🚗 Open http://localhost:5000 in your browser\n")
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)