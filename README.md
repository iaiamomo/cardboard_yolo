# Cardboard defects

## Experimental Results
*Experimental results are available in [validation](validation) folder.*

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

## Getting Started
0. Install [docker](https://www.docker.com/products/docker-desktop/)
1. Build the image
    ```bash
    docker build -t rotalaser-gpu .
    ```
2. Launch the container in detach mode with GPU acces. Be sure to have installed the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).
    ```bash
    docker run -d -v .:/home/default/ --gpus=all -it rotalaser-gpu bash
    ```
3. Open a terminal inside the created container (check the `<container_id>` with `docker ps`).
    ```bash
    docker exec -it <container_id> bash
    ```

## Training
*Training execution traces are available in [res](res) folder.*

If you want to train the models, follow the steps below:

0. Create an `.env` file containing the [ClearML](https://app.clear.ml) API keys.
1. Download the dataset from [roboflow](https://universe.roboflow.com/cardspace/hole_fold).
2. Extract the dataset into `dataset_det`
2. Run the scripts to convert the detection dataset into classification and segmentation
    ```bash
    cd dataset
    python3 det2cls.py
    python3 det2seg.py
    ```
3. To train the models you need to set up the [configuration file](src/config.json):
    ```json
    {
        "dim": "n",
        "device": 0,
        "train_id": 1
    }
    ```
    where:

    - "dim": represents the dimension of the model
    - "devide": represents the GPU on which running the training
    - "train_id": serves for tracing the log on ClearML, you can skip it
4. Run the following script for each train
    ```python
    cd src
    python3 yolov8_det.py   # YOLOv8 detection model
    python3 yolov8_cls.py   # YOLOv8 classification model
    python3 yolov8_seg.py   # YOLOv8 segmentation model
    ```

## Validation
1. Download the validation dataset from [roboflow](https://universe.roboflow.com/cardspace/cardboard_testset_).
2. Extract the dataset into `dataset_det_val`
2. Run the scripts to convert the detection dataset into classification and segmentation
    ```bash
    cd dataset
    python3 det2cls_val.py
    python3 det2seg_val.py
    ```
3. Run the following script to validate the various models 
    ```python
    cd src
    python3 yolov8_det_valid.py   # YOLOv8 detection model validation
    python3 yolov8_cls_valid.py   # YOLOv8 classification model validation
    python3 yolov8_seg_valid.py   # YOLOv8 segmentation model validation
    ```