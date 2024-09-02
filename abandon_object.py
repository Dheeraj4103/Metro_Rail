import cv2
import torch
import numpy as np
from ultralytics import YOLO
import pandas
import os
torch.cuda.set_device(0)

file = os.listdir(os.path.join(os.getcwd(),'videos1','Left_Object'))[0]
file = os.path.join(os.getcwd(),'videos1','Left_Object',file) 
model = YOLO("yolov9e.pt")

def predict(chosen_model, img, classes=[], conf=0.1):
    if classes:
        results = chosen_model.predict(img, classes=classes, conf=conf)
    else:
        results = chosen_model.predict(img, conf=conf)

    return results

def detection(results):
    inference = pandas.read_json(results[0].tojson())
    detection = []
    coords = inference['box']
    i = 0
    for coord in coords:
        x1, y1, x2, y2 = int(coord['x1']), int(coord['y1']), int(coord['x2']), int(coord['y2'])
        conf = inference['confidence'][i]
        class_idx = inference['class'][i]
        detection.append([x1, y1, x2, y2, conf, class_idx])
        i += 1
    return detection

from boxmot import DeepOCSORT
from pathlib import Path
import BagDetection

tracker = DeepOCSORT(
    model_weights=Path('osnet_x0_25_msmt17.pt'), # which ReID model to use
    device=torch.device("cuda:0"),
    fp16=False,
    
)


video_path = file  # Corrected path
cap = cv2.VideoCapture(video_path)

listOfObjects = []
frame_skip = 3
while cap.isOpened():
    for _ in range(frame_skip):
        ret, frame = cap.read()
    

    if not ret:
        break
    #frame = cv2.resize(frame, (1580,770), interpolation=cv2.INTER_AREA)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = predict(model,frame, classes=[0,24, 26, 28], conf=0.1)
    detections = detection(results)
    dets = np.array(detections)

    track = tracker.update(dets, frame)
    abandoned_objects = BagDetection.check_abandon(track, 50, list_of_bags=listOfObjects)
    print(abandoned_objects)
    if abandoned_objects[0]:
        
        cv2.putText(frame, f"Abandoned object detected", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1,
                (0, 0, 255), 2, cv2.LINE_AA)
        x1, y1, x2, y2 = abandoned_objects[2]
        cv2.rectangle(frame, (int(x1-15), int(y1-15)), (int(x2+15), int(y2+15)), (0, 0, 255), 4)
        
    print(track)
    # print(listOfObjects)
         # --> M X (x, y, x, y, id, conf, cls, ind)
    tracker.plot_results(frame, show_trajectories=False)
    
    cv2.imshow("video", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# writer.release()
cap.release()
cv2.destroyAllWindows()