import os
from ultralytics import YOLO
from clearml import Task
import gc
import torch
import dotenv
import time
dotenv.load_dotenv()

cwd = os.getcwd()

with open(f'{cwd}/outv8.txt', 'w') as f:
    print('YOLOv8 training', file=f)

batches = {
    'n': 32,
    's': 32,
    'm': 16
}
models = list(batches.keys())
datasets = ['dataset_fold' 'dataset_fold_negative', 'dataset_hole', 'dataset_hole_negative']
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
        task = Task.init(project_name='yolo', task_name=f"{dataset} - {dim} - {run_res}", tags=[dim, dataset, 'yolov8'])
        model = YOLO(f'yolov8{dim}.pt')
        model.train(data=f"{cwd}/{dataset}/data.yaml", epochs=500, batch=batches[dim], imgsz=640, patience=50, optimizer='SGD')
        task.close()
        #model = YOLO(f"{cwd}/yolov8/runs/detect/{run_res}/weights/best.pt")
        #results = model.predict(f"{cwd}/{dataset}/test/images/", save=True, imgsz=640, conf=0.7)
        
        gc.collect()
        torch.cuda.empty_cache()
        time.sleep(60)