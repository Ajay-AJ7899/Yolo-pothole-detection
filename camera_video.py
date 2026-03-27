# importing necessary libraries
import cv2 as cv
import time
import geocoder
import os

# Ensure the results directory exists to avoid save errors
result_path = "pothole_coordinates"
if not os.path.exists(result_path):
    os.makedirs(result_path)

# Reading label names from obj.names file
class_name = []
try:
    with open(os.path.join("project_files", "obj.names"), 'r') as f:
        class_name = [cname.strip() for cname in f.readlines()]
except FileNotFoundError:
    print("Error: 'project_files/obj.names' not found.")
    exit()

# Importing model weights and config file
net1 = cv.dnn.readNet('project_files/yolov4_tiny.weights', 'project_files/yolov4_tiny.cfg')
# net1.setPreferableBackend(cv.dnn.DNN_BACKEND_CUDA)
# net1.setPreferableTarget(cv.dnn.DNN_TARGET_CUDA_FP16)
# In your model parameters section, replace the CUDA lines with these:
net1.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
net1.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)

model1 = cv.dnn_DetectionModel(net1)
model1.setInputParams(size=(640, 480), scale=1/255, swapRB=True)

# Defining the video source (0 for camera or file name for video)
cap = cv.VideoCapture("pothole_video.mp4") 

# Safety check: Ensure video file or camera opened successfully
if not cap.isOpened():
    print("Error: Could not open video source 'pothole_video.mp4'. Check the file path.")
    exit()

# Get video properties for VideoWriter
width  = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
fps_val = 10

# Initialize VideoWriter with FFmpeg backend to solve the OpenCV exception
result = cv.VideoWriter('result.avi', 
                         cv.CAP_FFMPEG, # Specifically force FFmpeg backend
                         cv.VideoWriter_fourcc(*'MJPG'),
                         fps_val, (width, height))

# Defining initial values for parameters
g = geocoder.ip('me')
starting_time = time.time()
Conf_threshold = 0.5
NMS_threshold = 0.4
frame_counter = 0
i = 0
b = 0

# Detection loop
while True:
    ret, frame = cap.read()
    if not ret:
        print("End of video stream or cannot fetch frame.")
        break
        
    frame_counter += 1
    
    # Analysis of the stream with detection model
    classes, scores, boxes = model1.detect(frame, Conf_threshold, NMS_threshold)
    
    for (classid, score, box) in zip(classes, scores, boxes):
        label = "pothole"
        x, y, w, h = box
        recarea = w * h
        area = width * height
        
        # Drawing detection boxes and saving coordinates
        if len(scores) != 0 and score >= 0.7:
            if (recarea / area) <= 0.1 and y < 600:
                cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 1)
                cv.putText(frame, f"%{round(score*100, 2)} {label}", (x, y - 10), 
                            cv.FONT_HERSHEY_COMPLEX, 0.5, (255, 0, 0), 1)
                
                # Logic for saving pothole image and coordinates
                current_time = time.time()
                if i == 0 or (current_time - b) >= 2:
                    img_name = os.path.join(result_path, f'pothole{i}.jpg')
                    txt_name = os.path.join(result_path, f'pothole{i}.txt')
                    
                    cv.imwrite(img_name, frame)
                    with open(txt_name, 'w') as f:
                        f.write(str(g.latlng))
                    
                    b = current_time
                    i += 1

    # Calculating and displaying FPS
    elapsed_time = time.time() - starting_time
    fps = frame_counter / elapsed_time if elapsed_time > 0 else 0
    cv.putText(frame, f'FPS: {round(fps, 2)}', (20, 50),
               cv.FONT_HERSHEY_COMPLEX, 0.7, (0, 255, 0), 2)
    
    # Showing and saving result
    cv.imshow('Pothole Detection', frame)
    result.write(frame)
    
    if cv.waitKey(1) == ord('q'):
        break
    
# Release resources
cap.release()
result.release()
cv.destroyAllWindows()
