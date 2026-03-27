# Add at the top
from flask import Flask, Response, render_template
import threading
import cv2

# Your existing detection code here...

# Create Flask app
app = Flask(__name__)

# Modify your video processing loop to yield frames
def generate():
    while True:
        # Your frame processing code here
        # After drawing detections, encode frame
        ret, jpeg = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

# Run in a thread
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)