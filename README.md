# Cardboard defects

## Docker

```bash
docker build -t rotalaser-gpu .

docker run -p 8888:8888 -d -v /home/iseng/carboard_yolo:/home/default/ --runtime nvidia -it rotalaser-gpu bash

jupyter notebook --ip 0.0.0.0 --port 8888 --allow-root
```

### Jetson configuration

#### Torch
To install pytorch for jetson please follow this installation guide: https://forums.developer.nvidia.com/t/pytorch-for-jetson/72048

#### Torchvision
To install torchvision for jetson please follow this installation guide: https://forums.developer.nvidia.com/t/pytorch-for-jetson/72048 (Instructions > Installation > torchvision)


```python
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

## Yolo notes
- img size in train and detect: https://github.com/ultralytics/yolov5/issues/5851
- model with gab_dataset: https://files.clear.ml/YOLOv5/Training.a03a00ae52cd4acda82e26668bb96510/models/best.pt
