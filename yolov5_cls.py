import shutil
import os
from clearml import Task
import dotenv
dotenv.load_dotenv()

cwd = os.getcwd()

with open(f'{cwd}/outv5.txt', 'w') as f:
    print('YOLOv5 training', file=f)

batches = {
    'n': 64,
    's': 64,
    'm': 32
}
models = list(batches.keys())
datasets = ['classification/dataset_fold_cls']
idx = 1
for dataset in datasets:
    for dim in models:
        if idx == 1:
            run_res = "exp"
        else: run_res = f"exp{idx}"
        with open(f'{cwd}/outv5.txt', 'a') as f:
            print(f"{dataset} - YOLOv5{dim} - {run_res}", file=f)
        idx+=1
        
        os.chdir(f'{cwd}/yolov5')
        task = Task.init(project_name='yolo', task_name=f"{dataset} - {dim} - {run_res}", tags=[dim, dataset, 'yolov5'])
        res = os.system(f"python3 train.py --img 640 --batch {batches[dim]} --epochs 500 --patience 50 --optimizer 'SGD' --data data/data.yaml --weights yolov5{dim}.pt")
        task.close()
        #res = os.system(f"python3 detect.py --weights runs/train/{run_res}/weights/best.pt --img 640 --source ../{dataset}/test/images --line-thickness 1 --save-txt")