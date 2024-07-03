import os
from ultralytics import YOLO
from clearml import Task
import gc
import torch
import dotenv
import time
import json
dotenv.load_dotenv()

config = json.load(open('config.json'))
models = [config['dim']]
device_gpu = config['device']

cwd = os.getcwd()

with open(f'{cwd}/outv8.txt', 'w') as f:
    print('YOLOv8 training', file=f)

batches = {
    's': 16,
    'm': 16
}
#models = list(batches.keys())
datasets = ['dataset_all']
idx = 0
for dataset in datasets:
    for dim in models:
        if idx == 1:
            run_res = "train"
        else: run_res = f"train{idx}"
        with open(f'{cwd}/outv8.txt', 'a') as f:
            print(f"{dataset} - YOLOv8{dim} - {run_res}", file=f)
        idx+=1

        os.chdir(f'{cwd}/yolov8')
        task = Task.init(project_name='classification', task_name=f"{dataset} - {dim} - {run_res} - dgx", tags=[dim, dataset, 'yolov8'])
        model = YOLO(f'yolov8{dim}-cls.pt')
        model.train(data=f"{cwd}/classification/{dataset}", epochs=500, batch=batches[dim], imgsz=640, patience=30, optimizer='SGD', device=device_gpu)
        task.close()
        #model = YOLO(f"{cwd}/yolov8/runs/detect/{run_res}/weights/best.pt")
        #results = model.predict(f"{cwd}/{dataset}/test/images/", save=True, imgsz=640, conf=0.7)
        
        gc.collect()
        torch.cuda.empty_cache()
        time.sleep(60)