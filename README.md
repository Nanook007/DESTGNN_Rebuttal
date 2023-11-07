# DESTGNN_Rebuttal
This is a PyTorch implementation of the paper:Distinguishable Embedding Spatial-Temporal Graph Neural Network for Traffic Forecasting

# Requirements
The model is implemented using Python3.8.13, conda 4.5.4 , torch Version: 1.10.1+cu102

# Traffic datasets
You can download all the raw datasets at Google Drive https://drive.google.com/drive/folders/14EJVODCU48fGK0FkyeVom_9lETh80Yjp  or Baidu Yun(password: 6v0a)https://pan.baidu.com/share/init?surl=0gOPtlC9M4BEjx89VD1Vbw, and unzip them to datasets/raw_data/.

They should be downloaded to the code root dir and replace the raw_data and sensor_graph folder in the datasets

Alterbatively, the datasets can be found as follows:

PEMS04 and PEMS08: These datasets were released by ASTGCN[2] and ASTGNN[3]. Data can also be found in its GitHub repository.

# Model Training
python mian.py 

you can choose datasets by 

config_path = "configs/PEMS04.yaml"
config_path = "configs/PEMS08.yaml"
config_path = "configs/PEMS07.yaml"
