import os
import shutil
import dotenv
from clearml import Task

dotenv.load_dotenv()

shutil.copy('./data.yaml', './yolov5/data')

task = Task.init(project_name='yolov5_v3', task_name='dataset')

os.system("python3 yolov5/train.py --img 320 --batch 8 --epochs 100 --data data.yaml --weights yolov5s.pt --cache")

task.close()