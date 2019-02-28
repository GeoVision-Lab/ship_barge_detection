# Ship and Barge Detection from Satellite images
This readme describes methods to Create own object detection classifier using Large satellite images:
1. Installation
2. Gathering Satellite images and Labeled Json Data.
3. Cropping Large images and Labeled data to small batch
4. Generating training Data
5. Creating label maps and configuring training
6. Training
7. Exporting the inference graph
8. Testing newly trained object detection classifier


# Introduction 

The purpose of this tutorial is to explain how to train one's own object detection classifier especially using large remote sensing satellite images. At the end of this tutorial, you will have a methodology to properly convert large images and annotation data to training dataset as well as identify and draw boxes around small specific ojects in images. 

This tutorial is written using Windows 10 and tested on Linux as well. So, provided codes can work in both operating system.

# Process

## 1. Installation

* Download and Install anaconda with python 3.6 [link].
* Install tensorflow-gpu using anaconda (Simplest way of installing gpu version). No need to use cuda or cudnn files-
  
   ``` 
    conda install -c anaconda tensorflow-gpu 
    ```

* Install other necessary python packages. 
   ``` 
    conda install -c anaconda protobuf
    conda install -c anaconda pandas
    conda install -c anaconda pillow 
    conda install -c anaconda git
    ```
* Download this tutorial's repository from GitHub. 
  
  ```
   git clone https://github.com/UttamDwivedi/ship_barge_detection.git
  ``` 

## 2. Gathering Satellite images and Labeled Json Data

Follow the link to download raw data from [Tellus Satellite Challenge](https://signate.jp/competitions/153) or put your data on these folders. At the end your files should look like this: 

```
raw_data 
└───raw_train_images
│   │   train_00.jpg
│   │   train_01.jpg
│ 
└───raw_train_annotation
│   │   train_00.json
│   │   train_01.json
│ 
└───raw_test_images
    │   test_00.jpg
    │   test_00.jpg
```

## 3. Cropping Large images and Labeled data to small batch

Follow Jupyter notebook [train_image_label_split.ipynb](https://github.com/UttamDwivedi/ship_barge_detection/blob/master/train_image_label_split.ipynb)  to create training batch of 600 x 600 pixels. It divides large training images into small batch of training and validation dataset. The output will be saved in "training_data" folder. (Currently, the folder contains training dataset. To train on your own dataset, please remove them and add your own data.)  

The dataset required for training your own object detection classifier are following

```
training_data 
│
└───train_labels.csv
│
└───valid_labels.csv
│
└───train_images
│   │   train_00_03000_07800.jpg
│   │   train_01_03000_08400.jpg
│ 
└───valid_images
│   │   train_02_00000_00600.jpg
│   │   train_02_000600_01200.jpg
```

