import json
import time
import cv2
import numpy as np
from Video_Stream import video_stream
from shapely.geometry import Point, Polygon
from ultralytics import YOLO
import os
from deep_sort.utils.parser import get_config
from deep_sort.deep_sort import DeepSort
from deep_sort.sort.tracker import Tracker



person_thresh =20 #max number of person in queue

file = os.listdir(os.path.join(os.getcwd(),'videos','Queue'))[1]
file = os.path.join(os.getcwd(),'videos','Queue',file) 

deep_sort_weights = 'deep_sort/deep/checkpoint/ckpt.t7'
tracker = DeepSort(model_path=deep_sort_weights, max_age=50)
cap = cv2.VideoCapture(file)
output_path='./points_data/polygon_points.json'
img_path = './ref_img/queue_frame.jpg'
video_stream(file,img_path,output_path)
polygon_points=json.load(open('./points_data/polygon_points.json'))
polygon_points = np.array(polygon_points)
x_arr = polygon_points[:,0]
y_arr = polygon_points[:,1]
x_min = min(x_arr)
x_max = max(x_arr)
y_min = min(y_arr)
y_max = max(y_arr)

    # Define the coordinates for the region of interest (ROI) as a list of tuples
roi_points = [[0,0],*polygon_points,[0,0],[0,1080],[1920,1080],[1920,0],[0,0]]

i = 0
counter, fps, elapsed = 0, 0, 0
start_time = time.perf_counter()

    # Load YOLO model outside the loop
model = YOLO("yolov9c.pt")  # load a pretrained model (recommended for training)
frame_skip = 1
while cap.isOpened():
    for _ in range(frame_skip):
        ret, frame = cap.read()
            

    if ret:
        og_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        og_frame = cv2.resize(og_frame, (1580,770), interpolation=cv2.INTER_AREA)
            
            
            #og_frame = cv2.resize(og_frame,(1170,500))
        
        roi_poly = np.array([roi_points], dtype=np.int32)
        cv2.fillPoly(og_frame, roi_poly, (0,0,0))
        
            # Apply the mask to the frame
        og_frame = og_frame[y_min:y_max,x_min:x_max]

            # Perform object detection only within the ROI
        results = model(og_frame, classes=0, conf=0.2)
        class_names = ['person']
        for result in results:
            boxes = result.boxes  # Boxes object for bbox outputs
            probs = result.probs  # Class probabilities for classification outputs
            cls = boxes.cls.tolist()  # Convert tensor to list
            xyxy = boxes.xyxy
            conf = boxes.conf
            xywh = boxes.xywh  # box with xywh format, (N, 4)
            for class_index in cls:
                class_name = class_names[int(class_index)]
                    # print("Class:", class_name)

            pred_cls = np.array(cls)
            conf = conf.detach().cpu().numpy()
            xyxy = xyxy.detach().cpu().numpy()
            bboxes_xywh = xywh
            bboxes_xywh = xywh.cpu().numpy()
            bboxes_xywh = np.array(bboxes_xywh, dtype=float)

                # Convert bounding box coordinates from ROI frame to original frame
            bboxes_xywh[:, 0] += roi_points[0][0]
            bboxes_xywh[:, 1] += roi_points[0][1]
            bboxes_xywh[:, 2] += roi_points[0][0]
            bboxes_xywh[:, 3] += roi_points[0][1]

            # Perform tracking within the ROI
        tracks = tracker.update(bboxes_xywh, conf, og_frame)

        for track in tracker.tracker.tracks:
            track_id = track.track_id
            hits = track.hits
            x1, y1, x2, y2 = track.to_tlbr()  # Get bounding box coordinates in (x1, y1, x2, y2) format
            w = x2 - x1  # Calculate width
            h = y2 - y1  # Calculate height

                # Convert bounding box coordinates from ROI frame to original frame
            x1 += roi_points[0][0]
            y1 += roi_points[0][1]
            x2 += roi_points[0][0]
            y2 += roi_points[0][1]

            cv2.rectangle(og_frame, (int(x1), int(y1)), (int(x1 + w), int(y1 + h)), (0, 255, 0), 2)

            text_color = (0, 0, 0)  # Black color for text
            cv2.putText(og_frame, f"{class_name}-{track_id}", (int(x1) + 10, int(y1) - 5), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, text_color, 1, cv2.LINE_AA)

            # Add the track_id to the set of unique track IDs
        unique_track_ids = set()
        for track in tracker.tracker.tracks:
            unique_track_ids.add(track.track_id)

            # Update the person count based on the number of unique track IDs
        person_count = len(unique_track_ids)

            # Update FPS and place on frame
        current_time = time.perf_counter()
        elapsed = (current_time - start_time)
        counter += 1
        if elapsed > 1:
            fps = counter / elapsed
            counter = 0
            start_time = current_time

            # Draw person count on frame
        cv2.putText(og_frame, f"Person Count: {person_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (255, 255, 255), 2, cv2.LINE_AA)
        if person_count >= person_thresh:
            cv2.putText(og_frame, f"Maximum number has been reached", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0, 0, 255), 2, cv2.LINE_AA)

            # Show the frame
        cv2.imshow('video',og_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()