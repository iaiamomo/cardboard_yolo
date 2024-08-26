# DIE-VIS: an Automated Visual Inspection System for Cardboard Box Manufacturing

Repository containing the experimental results for the paper "DIE-VIS: an Automated Visual Inspection System for Cardboard Box Manufacturing" presented at the 2nd workshop on Vision-based InduStrial InspectiON (VISION) - ECCV 2024.

## Experimental Results
Experimental results are available in [validation](validation) folder. A `class_pr.txt` file is present in each folder reporting the *precision*, *accuracy* and other metrics. The structure of the [validation](validation) folder is as follows.
```
.
└── validation
    ├── classify                # classification task
    |   └── yolv8l-cls          # large model
    |   |   ├── class_pr.txt    # results
    |   |   └── ...
    |   └── ...                 # other models (x, m, s, n)
    ├── detect                  # detection task
    |   └── yolv8l-det          # large model
    |   |   ├── class_pr.txt    # results
    |   |   └── ...
    |   └── ...                 # other models (x, m, s, n)
    └── segment                 # segmentation task
        └── yolv8l-seg          # large model
        |   ├── class_pr.txt    # results
        |   └── ...
        └── ...                 # other models (x, m, s, n)
```

## Training details
Training execution traces are available in [res](res) folder. The structure of the [res](res) folder is as follows.
```
.
└── res
    ├── clearml_exp_cls.csv      # classification training details
    ├── clearml_exp_det.csv      # detection training details
    └── clearml_exp_seg.csv      # segmentation training details
```


## How to use the code
### Getting Started
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

### Training
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
    - "train_id": serves for tracing the log on ClearML. Remember to use an incremental number starting from 1 for each YOLOv8 task. It will be useful for tracking the `best.pt` model stored in the `training` folder of `yolov8`.
4. Run the following script for each train
    ```python
    cd src/training
    python3 yolov8_det.py   # YOLOv8 detection model
    python3 yolov8_cls.py   # YOLOv8 classification model
    python3 yolov8_seg.py   # YOLOv8 segmentation model
    ```

## Validation
1. Download the validation dataset from [roboflow](https://universe.roboflow.com/cardspace/cardboard_testset_).
2. Extract the dataset into `dataset_det_val`
2. Use the scripts to convert the detection dataset into classification and segmentation
4. Download the `clearml.csv` file from the tool to download the training info
3. Run the following script to validate the various models
    ```python
    cd src/validation
    python3 yolov8_det_valid.py   # YOLOv8 detection model validation
    python3 yolov8_cls_valid.py   # YOLOv8 classification model validation
    python3 yolov8_seg_valid.py   # YOLOv8 segmentation model validation
    ```

## License
Distributed under the MIT License. See [LICENSE](LICENSE) for more information.
