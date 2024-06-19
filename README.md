# Cardboard defects

## Getting Started
- Install [Miniconda](https://docs.anaconda.com/free/miniconda/) or [Anaconda](https://www.anaconda.com/download) if you haven't already.

- Create a new [conda](https://docs.anaconda.com/free/miniconda/) environment:
    ```bash
    conda create -n pyolo python=3.10
    conda activate pyolo
    ```

- Install the dependencies:
    ```bash
    pip install -r requirements.py
    ```

## Docker

1. Build the image
    ```bash
    docker build -t rotalaser-gpu .
    ```
2. Launch the container in detach mode with GPU acces. Be sure to have installed the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).
    ```bash
    # Jetson
    docker run -p 8888:8888 -d -v /home/iseng/cardboard_yolo:/home/default/ --runtime nvidia -it rotalaser-gpu bash

    # Server
    docker run -p 8888:8888 -d -v /home/iseng/cardboard_yolo:/home/default/ --gpus=all -it rotalaser-gpu bash
    ```
3. Open a terminal inside the created container (check the `<container_id>` with `docker ps`).
    ```bash
    docker exec -it <container_id> bash
    ```
4. Launch Jupyter
    ```bash
    jupyter notebook --ip 0.0.0.0 --port 8888 --allow-root
    ```

### Jetson configuration
These instructions are already applied in the creation of the container. They are here reported for information.

#### Torch
To install pytorch for jetson please follow this installation guide: https://forums.developer.nvidia.com/t/pytorch-for-jetson/72048

#### Torchvision
To install torchvision for jetson please follow this installation guide: https://forums.developer.nvidia.com/t/pytorch-for-jetson/72048 (Instructions > Installation > torchvision)

```bash
sudo apt-get install libjpeg-dev zlib1g-dev libpython3-dev libopenblas-dev libavcodec-dev libavformat-dev libswscale-dev
git clone --branch v0.15.1 https://github.com/pytorch/vision torchvision
cd torchvision
export BUILD_VERSION=0.15.1
python3 setup.py install --user
cd ../
pip install 'pillow<7'
```

#### Versions
- Version: JetPack 5.1 (L4T R35.2.1) / JetPack 5.1.1 (L4T R35.3.1) - Python 3.8 - torch-2.0.0+nv23.05-cp38-cp38-linux_aarch64.whl 5.0k
- PyTorch v2.0 - torchvision v0.15.1

## Yolov5
```bash
git clone https://github.com/ultralytics/yolov5.git
cd yolov5
pip install -qr requirements.txt
```

## Yolov8
```bash
from ultralytics import YOLO
model = YOLO('yolov8m.pt')
```

## Yolo notes
- img size in train and detect: https://github.com/ultralytics/yolov5/issues/5851
- model with gab_dataset: https://files.clear.ml/YOLOv5/Training.a03a00ae52cd4acda82e26668bb96510/models/best.pt


## Experiments

### Folds (dataset_f)
Dataset with:
- Negative examples
- Preprocessing: resize 640 and auto-orient
- Augmentation: rotation 90

#### Models
- YOLOv5 (epoch 400, patience 50)
    - s: (batch 64)
        - yolov5/runs/train/exp4
        - yolov5/runs/detect/exp5
    - m: (batch 32) 
        - yolov5/runs/train/exp6 non ha early stoppato
        - yolov5/runs/detect/exp6
- YOLOv8 (batch 32, epoch 400, patience 50)
    - s: yolov8/runs/detect/train12
        - yolov8/runs/detect/predict3
    - m: yolov8/runs/detect/train13
        - yolov8/runs/detect/predict4

### Holes (dataset_h)
Dataset with:
- Negative examples
- Preprocessing: resize 640 and auto-orient
- Augmentation: rotation 90

#### Models
- YOLOv5 (epoch 400, patience 50)
    - s: (batch 64)
        - yolov5/runs/train/exp7
        - yolov5/runs/detect/exp7
    - m: (batch 32) 
        - yolov5/runs/train/exp9 non ha early stoppato
        - yolov5/runs/detect/exp9
- YOLOv8 (batch 32, epoch 400, patience 50)
    - s: yolov8/runs/detect/train
        - yolov8/runs/detect/predict2
    - m: yolov8/runs/detect/train3 non ha early stoppato
        - yolov8/runs/detect/predict3