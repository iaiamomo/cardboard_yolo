import os
import cv2
import glob
import shutil
import random

fold_class = 0
hole_class = 1
data_d = "dataset_det"
data_c = "dataset_cls"
data_c_temp = "dataset_cls_temp"

os.mkdir(f"../{data_c_temp}")
os.mkdir(f"../{data_c_temp}/fold")
os.mkdir(f"../{data_c_temp}/hole")

all_images = glob.glob(f"../{data_d}/train/images/*.jpg")
for elem in all_images:
    img_name = elem.split("/")[3].split(".")[0]

    img = cv2.imread(elem)
    h, w = img.shape[:2]

    label_text = glob.glob(f"../{data_d}/train/labels/{img_name}.rf.*.txt")[0]
    with open(label_text, mode="r") as f:
        labels_lines = f.readlines()

    idx = 0
    for label in labels_lines:
        label_tokens = label.split(" ")

        # class
        class_label = int(label_tokens[0])

        # labels coordinates
        w_bbox = int(float(label_tokens[3]) * w)
        h_bbox = int(float(label_tokens[4]) * h)
        cx_bbox = int(float(label_tokens[1]) * w) - w_bbox // 2
        cy_bbox = int(float(label_tokens[2]) * h) - h_bbox // 2

        # img info extracted
        bbox = img[cy_bbox:cy_bbox+h_bbox, cx_bbox:cx_bbox+w_bbox]

        if class_label == fold_class:
            cv2.imwrite(f"../{data_c_temp}/fold/{img_name}_{idx}.jpg", bbox)
        else:
            cv2.imwrite(f"../{data_c_temp}/hole/{img_name}_{idx}.jpg", bbox)
        
        idx+=1

os.mkdir(f"../{data_c}")
os.mkdir(f"../{data_c}/train")
os.mkdir(f"../{data_c}/train/fold")
os.mkdir(f"../{data_c}/train/hole")
os.mkdir(f"../{data_c}/test")
os.mkdir(f"../{data_c}/test/fold")
os.mkdir(f"../{data_c}/test/hole")
os.mkdir(f"../{data_c}/val")
os.mkdir(f"../{data_c}/val/fold")
os.mkdir(f"../{data_c}/val/hole")

for type_d in ["fold", "hole"]:
    all_images = glob.glob(f"../{data_c_temp}/{type_d}/*.jpg")
    random.shuffle(all_images)
    n_elem = len(all_images)
    n_train = int(n_elem * 0.7)+2
    n_valid = int(n_elem * 0.2)
    n_test = int(n_elem * 0.1)
    images_train = all_images[:n_train]
    images_valid = all_images[n_train:n_train+n_valid]
    images_test = all_images[n_train+n_valid:]

    for elem in images_train:
        img_name = elem.split("/")[-1]
        shutil.copyfile(elem, f"{data_c}/train/{type_d}/{img_name}")

    for elem in images_test:
        img_name = elem.split("/")[-1]
        shutil.copyfile(elem, f"{data_c}/test/{type_d}/{img_name}")

    for elem in images_valid:
        img_name = elem.split("/")[-1]
        shutil.copyfile(elem, f"{data_c}/val/{type_d}/{img_name}")

shutil.rmtree(f"../{data_c_temp}")