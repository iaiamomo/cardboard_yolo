import os
import cv2
import glob
import shutil

fold_class = 0
hole_class = 1
data_d = "dataset_det"
data_c = "dataset_cls"

os.mkdir(f"../{data_c}")
os.mkdir(f"../{data_c}/train")
os.mkdir(f"../{data_c}/test")
os.mkdir(f"../{data_c}/val")
os.mkdir(f"../{data_c}/train/images")
os.mkdir(f"../{data_c}/train/labels")
os.mkdir(f"../{data_c}/test/images")
os.mkdir(f"../{data_c}/test/labels")
os.mkdir(f"../{data_c}/val/images")
os.mkdir(f"../{data_c}/val/labels")

d_sets = ["train", "test", "val"]
for d_set in d_sets:
    all_images = glob.glob(f"../{data_d}/{d_set}/images/*.jpg")
    for elem in all_images:
        img_name = elem.split("/")[-1].split(".")[0]

        img = cv2.imread(elem)
        h, w = img.shape[:2]

        label_text = glob.glob(f"../{data_d}/{d_set}/labels/{img_name}.rf.*.txt")
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

        with open(f"../{data_c}/{d_set}/labels/{img_name}.txt", "w") as f:
            f.write(new_labels)
        shutil.copyfile(elem, f"{data_c}/{d_set}/images/{img_name}.jpg")