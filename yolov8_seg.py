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
datasets = [config['dataset']]
idx = config['train_id']

cwd = os.getcwd()

batches = {
    'n': 32,
    's': 32,
    'm': 32,
    'l': 32,
    'x': 32
}
#models = list(batches.keys())
datasets = ['data_segmentation']
#idx = 7
for dataset in datasets:
    for dim in models:
        if idx == 1:
            run_res = "train"
        else: run_res = f"train{idx}"
        idx+=1

        os.chdir(f'{cwd}/yolov8')
        task = Task.init(project_name='segmentation', task_name=f"{dataset} - {dim} - {run_res} - dgx", tags=[dim, dataset, 'yolov8'])
        model = YOLO(f'yolov8{dim}-seg.pt')
        model.train(data=f"{cwd}/{dataset}/data.yaml", epochs=500, batch=batches[dim], imgsz=640, patience=30, optimizer='SGD', device=device_gpu)
        task.close()
        #model = YOLO(f"{cwd}/yolov8/runs/detect/{run_res}/weights/best.pt")
        #results = model.predict(f"{cwd}/{dataset}/test/images/", save=True, imgsz=640, conf=0.7)
        
        gc.collect()
        torch.cuda.empty_cache()
        time.sleep(60)
