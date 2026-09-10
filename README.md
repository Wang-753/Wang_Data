# HSM-YOLO: Sheep Mounting Behavior Detection

#####Enviornment   

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

####Dataset 
The dataset was collected from a commercial farm in Hangzhou. It can be obtained from ultralytics-main-v11\data\images. As the related funded project is still ongoing and due to the commercial confidentiality requirements of the farm, the complete dataset remains protected. At this stage, only a subset of the dataset has been released.


####Weigh 
The best.pt and last.pt weights obtained after training on this dataset are provided.
