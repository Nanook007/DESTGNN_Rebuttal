# DESTGNN_Rebuttal
This is a PyTorch implementation of the paper:Distinguishable Embedding Spatial-Temporal Graph Neural Network for Traffic Forecasting

# Requirements
The model is implemented using Python3.8.13, conda 4.5.4 , torch Version: 1.10.1+cu102

# Traffic datasets
Download the PEMS04 PEMS08 PEMS07 dataset from Google Drive or Baidu Yun provided by Li et al. . Move them into the dataset folder.

# Model Training
python mian.py 

you can choose datasets by 

config_path = "configs/PEMS04.yaml"
config_path = "configs/PEMS08.yaml"
config_path = "configs/PEMS07.yaml"
