import random
import glob
import shutil
data_c = "data_segmentation"

all_images = glob.glob(f"{data_c}/train/images/*.jpg")
random.shuffle(all_images)
n_elem = len(all_images) 
print(n_elem)
n_train = int(n_elem * 0.7)+2
n_valid = int(n_elem * 0.2)
n_test = int(n_elem * 0.1)
print(n_train)
print(n_valid)
print(n_test)
images_train = all_images[:n_train]
print(len(images_train))
images_valid = all_images[n_train:n_train+n_valid]
print(len(images_valid))
images_test = all_images[n_train+n_valid:]
print(len(images_test))

for elem in images_train:
    img_name = elem.split("/")[-1].split(".")[0]
    shutil.copyfile(elem, f"{data_c}/train2/images/{img_name}.jpg")
    shutil.copyfile(f"{data_c}/train/labels/{img_name}.txt", f"{data_c}/train2/labels/{img_name}.txt")

for elem in images_test:
    img_name = elem.split("/")[-1].split(".")[0]
    shutil.copyfile(elem, f"{data_c}/test/images/{img_name}.jpg")
    shutil.copyfile(f"{data_c}/train/labels/{img_name}.txt", f"{data_c}/test/labels/{img_name}.txt")

for elem in images_valid:
    img_name = elem.split("/")[-1].split(".")[0]
    shutil.copyfile(elem, f"{data_c}/val/images/{img_name}.jpg")
    shutil.copyfile(f"{data_c}/train/labels/{img_name}.txt", f"{data_c}/val/labels/{img_name}.txt")
