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

file = os.listdir(os.path.join(os.getcwd(),'videos','Platform'))[0]
file = os.path.join(os.getcwd(),'videos','Platform',file) 

cap = cv2.VideoCapture(file)
i = 0
counter, fps, elapsed = 0, 0, 0
start_time = time.perf_counter()

output_path='./points_data/edge_points.json'
img_path = './ref_img/edge_img.jpg'

video_stream(file,img_path,output_path)
edge_points=json.load(open(output_path))
    # Load YOLO model outside the loop
model = YOLO("yolov9c.pt")  # load a pretrained model (recommended for training)
frame_skip = 1
while cap.isOpened():
        
    for _ in range(frame_skip):
        ret, frame = cap.read()
        

    if ret:
        og_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        og_frame = cv2.resize(og_frame, (1580,770), interpolation=cv2.INTER_AREA)
        cv2.polylines(og_frame,np.array([edge_points]),False,(0,0,255,) if json.load(open('./points_data/is_inside.json')) else (0,0,0,),2)
        # Perform object detection only within the ROI
        results = model(og_frame,classes=[0,6], conf=0.4)
        class_names = ['person','bicycle','car','motorcycle','airplane','bus','train']
        for result in results:
            boxes = result.boxes  # Boxes object for bbox outputs
            probs = result.probs  # Class probabilities for classification outputs
            cls = boxes.cls.tolist()  # Convert tensor to list
            xyxy = boxes.xyxy
            conf = boxes.conf
            xywh = boxes.xywh  # box with xywh format, (N, 4)
            for class_index in cls:
                class_name = class_names[int(class_index)]
                    #print("Class:", class_name)

            pred_cls = np.array(cls)
            conf = conf.detach().cpu().numpy()
            xyxy = xyxy.detach().cpu().numpy()
            bboxes_xyxy = np.array(xyxy, dtype=float)
            bboxes_xywh = xywh
            bboxes_xywh = xywh.cpu().numpy()
            bboxes_xywh = np.array(bboxes_xywh, dtype=float)

            for bbox in bboxes_xyxy:
                x1 = bbox[0] 
                y1 = bbox[1] 
                x2 = bbox[2]
                y2 = bbox[3]
                point1 = Point(x2,y2)
                point2 = Point(x1,y2)
                polygon = Polygon(edge_points)
                is_inside=point1.within(polygon) or point2.within(polygon)
               
                if not json.load(open('./points_data/is_inside.json')) and  is_inside and 6 not in pred_cls:
                    with open('./points_data/is_inside.json', 'w') as f:
                        json.dump(is_inside, f)

                # Perform tracking within the ROI
                cv2.rectangle(og_frame, (int(x1), int(y2)), (int(x2), int(y1)), (0, 255, 0),2)

                text_color = (0, 0, 0)  # Black color for text
                cv2.putText(og_frame, f"{class_name}", (int(x1) + 10, int(y1) - 5), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, text_color, 1, cv2.LINE_AA)
            
            person_count = len(bboxes_xywh)
       

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
                (0, 0, 0), 2, cv2.LINE_AA)
        if json.load(open('./points_data/is_inside.json')):
            cv2.putText(og_frame, f"Someone has crossed the line", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1,
                (0, 0, 255), 2, cv2.LINE_AA)

        # Show the frame
        cv2.imshow("Video", og_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
           
with open('./points_data/is_inside.json', 'w') as f:
            json.dump(False, f)
cap.release()

cv2.destroyAllWindows()