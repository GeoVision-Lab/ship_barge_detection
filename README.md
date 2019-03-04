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

To get the basic idea of the model, please go through the tutorial jupyter notebook [ship_barge_detection_tutorial.ipynb](https://github.com/UttamDwivedi/ship_barge_detection/blob/master/ship_barge_detection_tutorial.ipynb). If you want to train your own classifier, please go through the process described below. 



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

## 5. Create label maps and configure training

### 5.1 Create label maps

The label map tells the trainer what each object is by defining a mapping of class names to class ID numbers. To train your own data replace id names to your own classes in the [labelmap.pbtxt](https://github.com/UttamDwivedi/ship_barge_detection/blob/master/training/labelmap.pbtxt) as following:

```
item {
  id: 1
  name: 'ship'
}

item {
  id: 2
  name: 'barge'
}
item {
  id: 3
  name: 'unknown'
}
```
### Changes required in configuration files

Finally, configure the object detection training pipeline in the [training](https://github.com/UttamDwivedi/ship_barge_detection/tree/master/training) folder. 
To train the model on your own dataset, please change in the following lines of [faster_rcnn_resnet101_sat_img.config](https://github.com/UttamDwivedi/ship_barge_detection/blob/master/training/faster_rcnn_resnet101_sat_img.config): 

* Line 4. Change num_classes to the number of different objects you want the classifier to detect. For the above ship, barge, and unknown, it would be num_classes : 3 .

* Line 18 and 20. Change the scale and aspect ratio of first stage anchor generator if the objects to be detected are very small or very large. 

* Line 104 and 105. uncomment these lines if you want to train the model with pretrained weights. (Be careful with the file path)

* Line 119. Change num_examples to the number of images you have in the \training_data\valid_images directory.

* Lines 114, 116, 125 and 127. Change the path if you have change the file names. 

Save the file after the changes have been made. That’s it! The training job is all configured and ready to go!

## 6. Begin training

From \ship_barge_detection folder, run the following command to start the magic: 

```
python train.py --logtostderr --train_dir=training/ --pipeline_config_path=training/faster_rcnn_resnet101_sat_img.config
```

Training will start after some initialization. Now go out and enjoy the nature for few hours.

### TensorBoard

Use TensorBoard to see the progres of training with some nice curves. To run this, go to terminal (command prompt in windows) and direct to ship_barge_detection folder and run the following command: 

```
# From ship_barge_detection folder
tensorboard --logdir=training
```

Now open your browser (e.g. chrome) and type "localhost:6006". The TensorBoard page shows a lot of information and graphs in regards to the progress of training. One important graph is Loss graph, which shows the overall loss of the classifier over time. 


### To stop the training, go to running command prompt window and terminate the training by pressing Ctrl+C. 
The training routine periodically saves checkpoints about every five minutes. So,  You can terminate training and start it later, and it will restart from the last saved checkpoint. The checkpoint at the highest number of steps will be used to generate the frozen inference graph.

## 7. Export Inference Graph

Now that training is complete, the last step is to generate the frozen inference graph (.pb file). From the \ship_barge_detection folder, issue the following command, where “XXXX” in “model.ckpt-XXXX” should be replaced with the highest-numbered .ckpt file in the training folder:

```
python export_inference_graph.py --input_type image_tensor --pipeline_config_path training/faster_rcnn_resnet101_sat_img.config --trained_checkpoint_prefix training/model.ckpt-XXXX --output_directory inference_graph
```

This creates a frozen_inference_graph.pb file in the \ship_barge_detection\inference_graph folder. The .pb file contains the object detection classifier.

## 8. Testing newly trained object detection classifier

Test the newly trained object detection classifier from [ship_barge_detection_tutorial.ipynb](https://github.com/UttamDwivedi/ship_barge_detection/blob/master/ship_barge_detection_tutorial.ipynb). 

* In the model Preparation part of the code, change the PATH_TO_FROZEN_GRAPH and NUM_CLASSES with the path of your brand new frozen_inference_graph.pb and number of classes of your object detection classifier. 

| Solarized dark             |  Solarized Ocean | 
| :-------------------------:|:-------------------------: | 
| ![](https://github.com/UttamDwivedi/ship_barge_detection/blob/master/results/test_images_output/image1.jpg)  |  ![](https://github.com/UttamDwivedi/ship_barge_detection/blob/master/results/test_images_output/image1.jpg)| 

![result](https://github.com/UttamDwivedi/ship_barge_detection/blob/master/results/test_images_output/image1.jpg)

![result](https://github.com/UttamDwivedi/ship_barge_detection/blob/master/results/test_images_output/image2.jpg)

![result](https://github.com/UttamDwivedi/ship_barge_detection/blob/master/results/test_images_output/image3.jpg)

![result](https://github.com/UttamDwivedi/ship_barge_detection/blob/master/results/test_images_output/image4.jpg)



