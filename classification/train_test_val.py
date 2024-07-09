import shutil
import random
import glob
import os

data_c = "dataset_classification"
type_c = "all"
type_d = "fold"

all_images = glob.glob(f"{data_c}/{type_d}/*.jpg")
random.shuffle(all_images)
n_elem = len(all_images) 
print(n_elem)
n_train = int(n_elem * 0.7)+2
n_valid = int(n_elem * 0.2)
n_test = int(n_elem * 0.1)
images_train = all_images[:n_train]
images_valid = all_images[n_train:n_train+n_valid]
images_test = all_images[n_train+n_valid:]
print(len(images_train)+len(images_valid)+len(images_test))

try:
    shutil.rmtree(f"dataset_{type_c}_cls/train/{type_d}")
except:
    print("prob")
os.mkdir(f"dataset_{type_c}_cls/train/{type_d}")

try:
    shutil.rmtree(f"dataset_{type_c}_cls/test/{type_d}")
except:
    print("prob")
os.mkdir(f"dataset_{type_c}_cls/test/{type_d}")

try:
    shutil.rmtree(f"dataset_{type_c}_cls/val/{type_d}")
except:
    print("prob")
os.mkdir(f"dataset_{type_c}_cls/val/{type_d}")

for elem in images_train:
    img_name = elem.split("/")[-1]
    shutil.copyfile(elem, f"dataset_{type_c}_cls/train/{type_d}/{img_name}")

for elem in images_test:
    img_name = elem.split("/")[-1]
    shutil.copyfile(elem, f"dataset_{type_c}_cls/test/{type_d}/{img_name}")

for elem in images_valid:
    img_name = elem.split("/")[-1]
    shutil.copyfile(elem, f"dataset_{type_c}_cls/val/{type_d}/{img_name}")
