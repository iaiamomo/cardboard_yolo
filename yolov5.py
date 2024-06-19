import shutil
import os
from clearml import Task
import dotenv
dotenv.load_dotenv()

models = ['n', 's', 'm']
datasets = ['dataset_f', 'dataset_h']
idx = 0
for dataset in datasets:
    shutil.copy(f'/home/calcio/cardboard_yolo/{dataset}/data.yaml', '/home/calcio/cardboard_yolo/yolov5/data')
    for dim in models:
        os.chdir('/home/calcio/cardboard_yolo/yolov5')
        task = Task.init(project_name='yolo', task_name=dim, tags=[dim, dataset, 'yolov5'])
        res = os.system("python3 train.py --img 640 --batch 32 --epochs 400 --patience 50 --optimizer 'SGD' --data data/data.yaml --weights yolov5m.pt")
        task.close()
        if idx == 0:
            run_res = "exp"
        else: run_res = f"exp{idx}"
        print(f"{dataset} - {dim} - {run_res}")
        idx+=1
        res = os.system(f"python3 detect.py --weights runs/train/{run_res}/weights/best.pt --img 640 --source ../{dataset}/test/images --line-thickness 1 --save-txt")