# Ship and Barge Detection from Satellite images
This readme describes methods to Create own object detection classifier using Large satellite images:
1. Installation
2. Gathering Satellite images and Labeled Json Data.
3. Cropping Large images and Labeled data to small batch
4. Generate training data as TFRecord
5. Create label maps and configure training
6. Begin training
7. Exporting the inference graph
8. Testing newly trained object detection classifier


# Introduction 

The purpose of this tutorial is to explain how to train one's own object detection classifier especially using large remote sensing satellite images. At the end of this tutorial, you will have a methodology to properly convert large images and annotation data to training dataset as well as identify and draw boxes around small specific ojects in images. 

This tutorial is written using Windows 10 and tested on Linux as well. So, provided codes can work in both operating system.

# Process

## 1. Installation

* Download and Install [Anaconda3-5.3.0](https://repo.continuum.io/archive/) with python 3.6 and make it default python. 
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

## 4. Generate training data as TFRecord 

TFRecord file format is a binary file format for storage of training and validation data. Binary data takes up less space on disk, takes less time to copy and can be read much more efficiently from disk. In case of datasets that are too large to be stored fully in memory, TFRecord file format is optimized for TensorFlow to load only the data that are required at the time (e.g. a batch).

To create TFRecord from our training data open [generate_tfrecored.py](https://github.com/UttamDwivedi/ship_barge_detection/blob/master/generate_tfrecord.py) and add your label map in the place of following lines. 

```
# TO-DO replace this with label map
def class_text_to_int(row_label):
    if row_label == 'ship':
        return 1
    elif row_label == 'barge':
        return 2
    elif row_label == 'unknown':
        return 3
    else:
        None
```
Now, generate the TFRecords files by running these commands from \ship_barge_detection folder

```
# For training data
python generate_tfrecord.py --csv_input=training_data\train_labels.csv --image_dir=training_data\train_images --output_path=train.record

# For validation data
python generate_tfrecord.py --csv_input=training_data\valid_labels.csv --image_dir=training_data\valid_images --output_path=valid.record
```
These generate a train.record and a valid.record file in \ship_barge_detection. These will be used to train the new object detection classifier.