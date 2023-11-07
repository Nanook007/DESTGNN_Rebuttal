# DESTGNN_Rebuttal
This is a PyTorch implementation of the paper:Distinguishable Embedding Spatial-Temporal Graph Neural Network for Traffic Forecasting

# Requirements
The model is implemented using Python3.8.13, conda 4.5.4 , torch Version: 1.10.1+cu102

# Traffic datasets
You can download all the raw datasets at Google Drive https://drive.google.com/drive/folders/14EJVODCU48fGK0FkyeVom_9lETh80Yjp  or Baidu Yun(password: 6v0a)https://pan.baidu.com/share/init?surl=0gOPtlC9M4BEjx89VD1Vbw, and unzip them to datasets/raw_data/.

# Data Preprocessing

cd /path/to/your/project
python scripts/data_preparation/${DATASET_NAME}/generate_training_data.py

Replace ${DATASET_NAME} with one of METR-LA, PEMS-BAY, PEMS03, PEMS04, PEMS07, PEMS08, or any other supported dataset. The processed data will be placed in datasets/${DATASET_NAME}.


# Model Training
python mian.py 

you can choose datasets by 

config_path = "configs/PEMS04.yaml"
config_path = "configs/PEMS08.yaml"
config_path = "configs/PEMS07.yaml"

# Experiment Results

![image](https://github.com/Nanook007/DESTGNN_Rebuttal/assets/84446048/751a591a-babc-41d3-a706-d1764a3ad6d8)

# The architecture and modules of DESTGNN

![image](https://github.com/Nanook007/DESTGNN_Rebuttal/assets/84446048/8470a75c-3b26-40d3-84a2-467d292ed985)


