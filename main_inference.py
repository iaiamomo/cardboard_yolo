import os

exp_dir = "exp9"
test_dir = "../dataset/images/test"

os.system(f"python yolov5/detect.py --weights yolov5/runs/train/{exp_dir}/weights/best.pt --img 320 --source {test_dir} --line-thickness 1 --save-txt")
