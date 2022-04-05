import roboflow
import json
import yolov5.train
import yolov5.detect_local
from pathlib import Path
import os

FILE = Path(__file__).resolve()
# print(FILE)
ROOT = FILE.parents[0]  # YOLOv5 root directory
# print(ROOT)

def run_ui_prediction(version_number):
    #This doesn't work because for some reason, project.versions() returns an empty list
    #Included for reference

    rf = roboflow.Roboflow(api_key="MY KEY") #TODO: Change this to personal API key if running
    # print(rf.workspace().projects())
    # print(rf.workspace())
    project = rf.workspace().project("duckhunt")
    print(project)

    #initial model trained by roboflow
    print(project.versions())
    model = project.version(version_number).model
    prediction = model.predict("./screenshots/manualtest/test1.png")
    prediction.plot()

    results = json.loads(prediction.json())
    coords = [(r['x'], r['y']) for r in results if r['type'] == 'duck']
    print(coords)

    return coords

def train_yolov5():
    # Usage: import train; train.run(data='coco128.yaml', imgsz=320, weights='yolov5m.pt')
    #trains yolov5s on the training data
    yolov5.train.run(data=ROOT / 'yolov5data/data.yaml', imgsz=640, weights='yolov5s.pt')

def predict_yolov5(frame):
    results = yolov5.detect_local.run(weights=ROOT / 'yolov5/runs/train/exp5/weights/best.pt', #source=ROOT / 'screenshots/manualtest2', 
                        data=ROOT / 'yolov5data/data.yaml', name='exp5', conf_thres=0.7, save_txt=True, save_conf=True, return_val=True, im=frame)
    x_y_coords = [(p[1], p[2]) for p in results if p[0] == 0]
    return x_y_coords

# train_yolov5()
# predict_yolov5()
# print(predict_yolov5())
