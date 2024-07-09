import os
import cv2
import time
import glob
import pickle
import shutil
import requests
import numpy as np
from datetime import datetime, timedelta
from dateutil import tz
import matplotlib.pyplot as plt

fold_class = 0
hole_class = 1
data_d = "dataset_all"
data_c = "data_segmentation"

os.mkdir(data_c)
os.mkdir(f"{data_c}/train")
os.mkdir(f"{data_c}/test")
os.mkdir(f"{data_c}/val")
os.mkdir(f"{data_c}/train/images")
os.mkdir(f"{data_c}/train/labels")
os.mkdir(f"{data_c}/test/images")
os.mkdir(f"{data_c}/test/labels")
os.mkdir(f"{data_c}/val/images")
os.mkdir(f"{data_c}/val/labels")

folder = "train"
path_ = f"../{data_d}/{folder}/images/*.jpg"
print(path_)
all_images = glob.glob(path_)
print(len(all_images))
for elem in all_images:
    img_name = elem.split("/")[-1].split(".")[0]
    print(img_name)

    img = cv2.imread(elem)
    h, w = img.shape[:2]

    label_text = glob.glob(f"../{data_d}/train/labels/{img_name}.rf.*.txt")
    
    print(label_text)
    label_text = label_text[0]
    with open(label_text, mode="r") as f:
        labels_lines = f.readlines()

    new_labels = ""
    for label in labels_lines:
        label_tokens = label.split(" ")

        # class
        class_label = int(label_tokens[0])

        # coordinates
        x, y, w, h = map(float, label_tokens[1:])
        x_min = x - (w / 2)
        y_min = y - (h / 2)
        x_max = x + (w / 2)
        y_max = y + (h / 2)

        new_labels += f"{class_label} {x_min:.6f} {y_min:.6f} {x_max:.6f} {y_min:.6f} {x_max:.6f} {y_max:.6f} {x_min:.6f} {y_max:.6f}\n"

    with open(f"{data_c}/{folder}/labels/{img_name}.txt", "w") as f:
        f.write(new_labels)
    shutil.copyfile(elem, f"{data_c}/{folder}/images/{img_name}.jpg")
