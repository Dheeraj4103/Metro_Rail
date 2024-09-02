import cv2
from polyplot import PolygonDrawer
import os
from pathlib import Path
import json

def draw_Boundary(img_path,output_path):
  cv2.imread(img_path)
  if os.path.getsize(output_path) == 0:
     pd = PolygonDrawer("Polygon")
     pd.run(img_path)
     with open(output_path, 'w') as f:
        json.dump(pd.points, f)


def video_stream(video_stream_url,img_path,output_path):
    cap = cv2.VideoCapture(video_stream_url)
    isFirst =True
    if not cap.isOpened():
        print("Error: Unable to open video stream.")
        exit()
    ret, frame = cap.read()

    if not ret:
        print("Error: Unable to read frame.")
        return
    cv2.imwrite(img_path,frame)
    draw_Boundary(img_path,output_path)

    cap.release()
    cv2.destroyAllWindows()
