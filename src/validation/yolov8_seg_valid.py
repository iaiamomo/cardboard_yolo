from ultralytics import YOLO
from clearml import Task
import pandas as pd
import os
import dotenv
dotenv.load_dotenv()

exp = pd.read_csv('clearml_exp_seg.csv')
exp_name = exp['NAME']
res_organized = {}
for i in range(len(exp_name)):
    tokens = exp_name[i].split(' - ')    
    dataset = tokens[0]
    dim_model = tokens[1]
    train_name = tokens[2]
    if dataset not in res_organized:
        res_organized[dataset] = []
    res_organized[dataset].append({
        "dim": dim_model,
        "train": train_name
    })

cwd = os.getcwd()

res_list = []
idx = 1
datasets = list(res_organized.keys())
for dataset in datasets:
    trains = res_organized[dataset]
    for train in trains:
        dim_model = train['dim']
        train_name = train['train']

        if idx == 1:
            run_res = "val"
        else: run_res = f"val{idx}"
        idx+=1

        os.chdir(f'{cwd}/yolov8')

        task = Task.init(project_name='validation', task_name=f"{dataset}-{dim_model}-{run_res}", tags=[dim_model, dataset, 'yolov8'])
        path_model = f'runs/segment/{train_name}/weights/best.pt'
        
        data_path = "/home/default/val_dataset/seg_test/data.yaml"

        model = YOLO(path_model)
        metrics = model.val(data=data_path, device="cuda:1", save_json=True)
        task.close()

        res = [dataset, dim_model, run_res, metrics.box.map, metrics.box.map50, metrics.box.map75, metrics.box.mp, metrics.box.mr, metrics.seg.map, metrics.seg.map50, metrics.seg.map75, metrics.seg.mp, metrics.seg.mr]
        res_list.append(res)

df = pd.DataFrame(res_list, columns=['DATASET', 'DIM_MODEL', 'RUN', 'B_MAP', 'B_MAP50', 'B_MAP75', 'B_MP', 'B_MR', 'S_MAP', 'S_MAP50', 'S_MAP75', 'S_MP', 'S_MR'])
df.to_csv('res_val_seg.csv', sep=',', index=False)