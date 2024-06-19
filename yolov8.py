import os
from ultralytics import YOLO
from clearml import Task
import dotenv
dotenv.load_dotenv()

models = ['n', 's', 'm']
datasets = ['dataset_f', 'dataset_h']
idx = 0
for dataset in datasets:
    for dim in models:
        os.chdir('/home/calcio/cardboard_yolo/yolov8')
        task = Task.init(project_name='yolo', task_name=dim, tags=[dim, dataset, 'yolov8'])
        model = YOLO(f'yolov8{dim}.pt')
        model.train(data=f"/home/calcio/cardboard_yolo/{dataset}/data.yaml", epochs=400, batch=32, imgsz=640, patience=50)
        task.close()
        if idx == 0:
            run_res = "train"
        else: run_res = f"train{idx}"
        print(f"{dataset} - {dim} - {run_res}")
        idx+=1
        model = YOLO(f"/home/calcio/cardboard_yolo/yolov8/runs/detect/{run_res}/weights/best.pt")
        results = model.predict(f"/home/calcio/cardboard_yolo/{dataset}/test/images/", save=True, imgsz=640, conf=0.7)