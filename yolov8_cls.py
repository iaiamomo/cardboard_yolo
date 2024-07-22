import os
from ultralytics import YOLO
from clearml import Task
import dotenv
import json
dotenv.load_dotenv()

config = json.load(open('config.json'))
models = [config['dim']]
device_gpu = config['device']
idx = config['train_id']

cwd = os.getcwd()

batches = {
    'n': 32,
    's': 32,
    'm': 32,
    'l': 32,
    'x': 32
}
datasets = ['dataset_all_cls']
for dataset in datasets:
    for dim in models:
        if idx == 1:
            run_res = "train"
        else: run_res = f"train{idx}"
        idx+=1

        os.chdir(f'{cwd}/yolov8')
        task = Task.init(project_name='classification', task_name=f"{dataset} - {dim} - {run_res} - dgx", tags=[dim, dataset, 'yolov8'])
        model = YOLO(f'yolov8{dim}-cls.pt')
        model.train(data=f"{cwd}/classification/{dataset}", epochs=500, batch=batches[dim], imgsz=640, patience=30, optimizer='SGD', device=device_gpu)
        task.close()
