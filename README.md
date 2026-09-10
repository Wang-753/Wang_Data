# HSM-YOLO: Hu Sheep Mounting Behavior Detection

## Environment
```bash
conda create -n pytorch python=3.7
conda activate pytorch
conda install -c pytorch pytorch=1.10.1 torchvision=0.11.2
pip install ultralytics
conda install matplotlib pandas tqdm
pip install opencv-python==4.11.0.86
pip install cython scipy
pip install pycocotools jpeg4py
pip install wget yacs
pip install shapely==1.6.4.post2
pip install numpy==2.0.2
```


## Quick Start
```
Place the downloaded sheep mounting behavior dataset or your own dataset in the ultralytics-main-v11/data;
Configure the relevant file paths required for running the YOLO model. Please refer to https://github.com/ultralytics/ultralytics;
Train the detection model: run ultralytics/models/yolo/detect/our_train.py.
```

## Dataset
```
The dataset is a sheep mounting behavior dataset collected from a commercial farm in Zhejiang Province. It can be obtained from ultralytics-main-v11\data\images. As the related funded project is still ongoing and due to the commercial confidentiality requirements of the farm, the complete dataset remains protected. At this stage, only a subset of the dataset has been released. Furthermore, the dataset contains multiple imaging scenarios. Non-commercial academic researchers who require access to the complete dataset may contact the corresponding author.


```

## Weigh 
The best.pt and last.pt weights obtained after training on this dataset are provided.


