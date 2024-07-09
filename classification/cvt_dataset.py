import os
import cv2
import glob
import shutil

fold_class = 0
hole_class = 1
data_d = "dataset_all"
data_c = "dataset_classification"

all_images = glob.glob(f"{data_d}/train/images/*.jpg")
for elem in all_images:
    img_name = elem.split("/")[3].split(".")[0]

    img = cv2.imread(elem)
    h, w = img.shape[:2]

    label_text = glob.glob(f"{data_d}/train/labels/{img_name}.rf.*.txt")[0]
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
            cv2.imwrite(f"{data_c}/fold/{img_name}_{idx}.jpg", bbox)
        else:
            cv2.imwrite(f"{data_c}/hole/{img_name}_{idx}.jpg", bbox)
        
        idx+=1
