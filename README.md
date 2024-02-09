# TorchEye

## This is an YOLOv3/v4 implementation, with support for training, inference and evaluation.

![](https://img.shields.io/static/v1?label=python&message=3.10&color=blue)
![](https://img.shields.io/static/v1?label=pytorch&message=2.0&color=<COLOR>)
[![](https://img.shields.io/static/v1?label=license&message=Apache2&color=green)](./License.txt)

```
├── README.md
├── autoDetect.py            detect objects in new image files in given folder
├── convert_weights.py       tool to convert *.pth checkpoint weight files --> darknet *.weights
├── detect.py                detect objects in images on given folder
├── detectVideo.py           detect objects in video file
├── models.py                model for pytorch
├── train.py                 train models
├── test.py                  test models
├── requirements.txt         requirements list
├── checkpoints              folder for last and best training checkpoints
│   ├── best                 folder for best training checkpoint
├── config
│   ├── autodetect.cfg       configuration file for autodetect.py
│   ├── clearml.cfg          configuration file for clearml
│   ├── create_custom_model.sh  bash script for creating a custom yolov3 config file
│   ├── custom.data          basefile for custom *.data file
│   ├── Test.data            datafile for testing
│   ├── Test.names           classfile for testing
│   ├── test_image.jpg       cuda test image for detectVideo.py
│   ├── Test-tiny.cfg        configuration file for testing
│   ├── yolov3.cgf           yolov3 configuration file
│   ├── yolov3-tiny.cfg      yolov3-tiny configuration file
├── data
│   ├── custom               folder structure for custom datasets
│   ├── samples              folder with sample files
│   ├── coco.names           coco class names
│   ├── get_coco_dataset     bash script for getting full coco dataset
│   ├── test_set.zip         ZIP file for testing
├── logs                     folder for training logging
├── output                   folder for detection output writing
├── utils                    folder with different tools
├── weights                  folder for weight files   
│   ├── download_weights.sh  bash script for downloading yolov3 darknet weights  

```


# Guide
## Installation

1. Unzip or pull package from repository into folder
2. Go to program folder
3. (Optional) If you want to use NVIDIA GPU acceleration, get suitable package from PyTorch [https://pytorch.org/get-started/locally/] (CUDA 11.8 or never) and install it
4. Type following command: 
```python
pip install -r requirements.txt
```
5. (Optional) If you want to use ClearML server, configure it by typing command:
```python
clearml-init
```
6. (Optional) If you want to use Backblaze storage, go to [/utils/b2-python-s3/] and type command:
```python
pip install -r requirements.txt
```

## 1.Dataset
### Collect your dataset, example sources:

* OIDv4 ToolKit - https://github.com/EscVM/OIDv4_ToolKit
* Imagenet - https://image-net.org/update-mar-11-2021.php
* Labelme - http://labelme.csail.mit.edu/Release3.0/browserTools/php/dataset.php
* Open images - https://blog.research.google/2016/09/introducing-open-images-dataset.html
* The Comprehensive Cars - http://mmlab.ie.cuhk.edu.hk/datasets/comp_cars/index.html
* VQA - https://visualqa.org


### Guides and other citations:
* What Data Quality Means to the Success of Your ML Models - https://imerit.net/blog/quality-labeled-data-all-pbm/
* Selecting Data Labeling Tools Doesn’t Have To Be Hard - https://imerit.net/blog/data-labeling-tools-avi-pbm/
* Training Yolo for Object Detection in PyTorch with Your Custom Dataset — The Simple Way - https://towardsdatascience.com/training-yolo-for-object-detection-in-pytorch-with-your-custom-dataset-the-simple-way-1aa6f56cf7d9
* Training YOLOV3: Deep Learning Based Custom Object Detection - https://learnopencv.com/training-yolov3-deep-learning-based-custom-object-detector/
## 2.Data annotation
Annotate your data, guides and software:
* Guide for tool selection - https://www.labellerr.com/blog/top-10-image-labeling-tool-of-2023/
* Label-studio - https://github.com/HumanSignal/label-studio or https://labelstud.io
* CVAT - https://www.cvat.ai/?ref=blog.roboflow.com
* MakeSense - https://www.makesense.ai/?ref=blog.roboflow.com
* LabelBox - https://labelbox.com/?ref=blog.roboflow.com
* Scale - https://scale.com/?ref=blog.roboflow.com
* VGG - https://github.com/nearkyh/via-1.0.5
* COCO Annotator - https://github.com/jsbroks/coco-annotator#readme
* VoTT - https://github.com/microsoft/VoTT/releases/tag/v2.2.0
* SuperAnnotate - https://www.superannotate.com
* 
## 3.Download the Pre-trained model
Go to https://pjreddie.com/darknet/yolo/ and download pre-trained YOLO model

## 4.Data and class files
If you're creating custom dataset create your own *.names file with each class as its own line
Then create your own datafile or make a copy of custom.data in config folder and modify that:
```python
classes = 1
train  = /path/to/snowman/snowman_train.txt
valid  = /path/to/snowman/snowman_test.txt
names = /path/to/snowman/classes.names
backup = /path/to/snowman/weights/
```
    
## 5.YOLOv3 configuration parameters
If you're creating your own dataset, then use 'create_custom_model.sh' on /config/ folder or make a copy existing 
yolov3.cfg or yolov3-tiny.cfg file and modify following lines:

* find first [yolo] section in config file, modify it's classes line (set number of classes you're going to train) and modify previous [convolutional] sections filters: filters = (classes + 5) x 3 
* then do the same for other yolo sections and convolutional sections before them

### 5.1 Batch hyper-parameter in training YOLOv3
```python
# Testing
# batch=1
# subdivisions=1
# Training
batch=64
subdivisions=16
```
The batch parameter indicates the batch size used during training.

Our training set can contain a few hundred images, but it is not uncommon to train on millions of images. The training process involves iteratively updating the weights of the neural network based on how many mistakes it is making on the training dataset.

It is impractical (and unnecessary) to use all images in training the set at once to update the weights. So, a small subset of images is used in one iteration, and this subset is called the batch size.

When the batch size is set to 64, it means 64 images are used in one iteration to update the parameters of the neural network.

NOTE! This is just for information only -> the software uses an autobatcher for calculating optimal batch size based on GPU memory available.
### 5.2 Subdivisions configuration parameter in training YOLOv3
Darknet allows you to specify a variable called subdivisions that lets you process a fraction of the batch size at one time on your GPU.

You can start the training with subdivisions=1, and if you get an Out of memory error, increase the subdivision parameter by multiples of 2(e.g. 2, 4, 8, 16) till the training proceeds successfully. The GPU will process $batch/subdivision$ number of images at any time, but the full batch or iteration would be complete only after all the images in batch (set above) are processed.
NOTE! This is just for information only -> the software uses an autobatcher for calculating optimal batch size based on GPU memory available.
### 5.3 Width, Height, Channels
These configuration parameters specify the input image size and the number of channels.
```python
width=416
height=416
channels=3
```
The input training images are first resized to width x height before training. Here we use the default values of 416×416. The results might improve if we increase it to 608×608, but it would take longer to train. channels=3 indicates that we would be processing 3-channel RGB input images.
### 5.4 Momentum and Decay
The configuration file contains a few parameters that control how the weight is updated.
```python
momentum=0.9
decay=0.0005
```

In the previous section, we mentioned how the weights of a neural network are updated based on a small batch of images and not the entire dataset. 
Because of this reason, the weight updates fluctuate quite a bit. That is why a parameter momentum is used to penalize large weight changes between iterations.

A typical neural network has millions of weights, and therefore it can easily overfit any training data. 
Overfitting simply means it will do very well on training data and poorly on test data. 
It is almost like the neural network has memorized the answer to all images in the training set but really not learned the underlying concept. 
One of the ways to mitigate this problem is to penalize large values for weights. The parameter decay controls this penalty term. 
The default value works just fine, but you may want to tweak this if you notice overfitting.
### 5.5 Learning Rate, Steps, Scales, Burn In (warm-up), Optimizers and Learning rate schedulers
#### Learning Rate, Steps, Scales
```python
# Number of warmup epochs, usually should not be more than 3
warmup=1
...
# initial learning rate
lr0=0.01
lrf=0.1
learning_rate=0.001
burn_in=2000
max_batches = 500200
policy=steps
steps=5000,50000,400000,450000
scales=.2,.2,.1,.1
```

The parameter learning rate controls how aggressively we should learn based on the current batch of data. 
Typically this is a number between 0.01 and 0.0001.

At the beginning of the training process, we start with zero information and so the learning rate needs to be high. 
But as the neural network sees a lot of data, the weights need to change less aggressively. 
In other words, the learning rate needs to be decreased over time. 
In the configuration file, this decrease in learning rate is accomplished by first specifying that our learning rate decreasing policy is steps. 
In the above example, the learning rate will start from 0.001 and remain constant for 5000 iterations, and then it will multiply by scales to get the new learning rate. 
We have also specified multiple steps and scales.

In the previous paragraph, we mentioned that the learning rate needs to be high initially and low later on. 
While that statement is largely true, it has been empirically found that the training speed tends to increase if we have a lower learning rate for a short period of time at the very beginning. This is controlled by the burn_in parameter. Sometimes this burn-in period is also called warm-up period.

NOTE! The warmup phase is automatically calculated and only parameter in config file that affects its length is 'warmup'
#### Burn In (warm-up)
```python
warmup_epochs = int(model.hyperparams['warmup'])
num_batches = len(dataloader)  # number of batches
warmup_num = max(
        round(warmup_epochs * num_batches), 100
    )  # number of warmup iterations, max(3 epochs, 100 iterations)
```
#### Optimizers and schedulers
Short description about both
##### AdamW
AdamW is a variant of Adam optimizer which corrects the weight decay. 
Weight decay is a regularization method where a small fixed proportion of the weights is subtracted at each training step. 
AdamW is often preferred to original Adam because of its handling of weight decay that leads to better generalization and therefore better performance.
##### SGD (Stochastic Gradient Descent)
This is the basis of many other optimizers in machine learning. 
It updates the parameters by directly subtracting the gradient of the loss function for a single training sample. 
It's simple and computationally efficient, but it can be much slower to converge than more modern optimizers.

##### RMSProp (Root Mean Square Propagation) 
RMSProp is known for its effective handling of the diminishing learning rates. 
It uses a moving average of squared gradients to normalize the gradient.

##### Adadelta 
Adadelta is an extension of Adagrad that seeks to reduce its aggressive, monotonically decreasing learning rate. 
It dynamically adapts over time using only the raw gradient information and has minimal computational overhead.

##### Adamax 
Adamax is a variant of Adam. 
It's considered to be more robust to noise as it uses the max norm of the gradients, 
making it suitable for tasks where the parameters need robust updating. 

--------------------------------------------------------------------------------------------------------------------

##### CosineAnnealingLR
This scheduler adjusts the learning rate using a cosine annealing schedule. 
It decreases the learning rate from the maximum to the minimum according to a cosine function. 
After reaching the minimum learning rate, it restarts in a cyclic manner.

![cosineannealing_lr.png](images%2Fcosineannealing_lr.png)

##### ChainedScheduler
A scheduler that combines multiple learning rate schedulers in a sequence.

##### ExponentialLR
ExponentialLR decreases the learning rate for each epoch with a constant rate. 
This useful when you want to reduce learning rate gradually.

![exponential_lr.png](images%2Fexponential_lr.png)

##### ReduceLROnPlateau 
This scheduler reduces learning rate when a metric has stopped improving. 
This is effective when the model hits a plateau in learning and an adjustment in the learning rate can stimulate more 
learning or fine-tuning.

##### ConstantLR
This scheduler keeps the learning rate constant for all epochs. 
It’s useful for fine-tuning models when we don't want the learning rate to change.

##### CyclicLR
This scheduler varies the learning rate between two boundaries with a cyclic schedule. 
This is beneficial when we're unsure about how small or large the learning rate should be.

![cyclic_lr_exp_range.png](images%2Fcyclic_lr_exp_range.png)
![cyclic_lr_triangular.png](images%2Fcyclic_lr_triangular.png)
![cyclic_lr_triangular2.png](images%2Fcyclic_lr_triangular2.png)
##### OneCycleLR
The scheduler adjusts the learning rate according to the 1cycle policy. 
It starts from a lower learning rate and gradually reaches the maximum learning rate. 
Post that, it starts decreasing the learning rate slowly. 
This schedule is generally used in training wide residual networks.

![onecycle_lr_cos.png](images%2Fonecycle_lr_cos.png)
![onecycle_lr_linear.png](images%2Fonecycle_lr_linear.png)

##### LambdaLR
With LambaLR, you can pass any function to define the learning rate adjustment.
This makes this scheduler highly flexible.

![lambda_lr.png](images%2Flambda_lr.png)

##### Multistep

![multistep_lr.png](images%2Fmultistep_lr.png)

##### Step

![step_lr.png](images%2Fstep_lr.png)

##### Multiplicative

![multiplicative_lr.png](images%2Fmultiplicative_lr.png)

##### CosineAnnealingWarmRestrats

![cosineannealingwarmrestarts.png](images%2Fcosineannealingwarmrestarts.png)


##### Valid optimizer - scheduler combinations
| Scheduler → Optimizer ↓ | CosineAnnealingLR (Scheduler test1) | ChainedScheduler (Scheduler test2) | ExponentialLR (Scheduler test3) | ReduceLROnPlateau (Scheduler test4) | ConstantLR (Scheduler test5) | CyclicLR (Scheduler test6) | OneCycleLR (Scheduler test7) | LambdaLR (Scheduler test8) | MultiplicativeLR | StepLR | MultiStepLR | LinearLR | PolynomialLR | CosineAnnealingWarmRestarts |
|-------------------------|-------------------------------------|------------------------------------|---------------------------------|-------------------------------------|------------------------------|----------------------------|------------------------------|----------------------------|------------------|--------|-------------|----------|--------------|-----------------------------|
| adamw                   | x                                   | x                                  | x                               | x                                   | x                            | -                          | x                            | x                          | x                | x      | x           | x        | x            | x                           |
| sgd                     | x                                   | x                                  | x                               | x                                   | x                            | x                          | x                            | x                          | x                | x      | x           | x        | x            | x                           |
| rmsprop                 |                                     |                                    |                                 |                                     |                              |                            |                              |                            |                  |        |             |          |              |                             |
| adam                    | x                                   | x                                  | x                               | x                                   | x                            | -                          | x                            | x                          | x                | x      | x           | x        | x            | x                           |
| adadelta                |                                     |                                    |                                 |                                     |                              |                            |                              |                            |                  |        |             |          |              |                             |
| adamax                  |                                     |                                    |                                 |                                     |                              |                            |                              |                            |                  |        |             |          |              | x                           |
### 5.6 Data augmentation
Data collection and annotation might take a long time.

So it would be good idea to maximize data by cooking up new data. 
This process is called data augmentation. 
For example, an image of a car is rotated by 5 degrees is still an image of a car. 
The angle parameter in the configuration file allows you to randomly rotate the given image by ± angle.

Similarly, if we transform the colors of the entire picture using saturation, exposure, and hue, 
it is still a picture of a car.
```python
angle=0
saturation = 1.5
exposure = 1.5
hue=.1
```
These values can be set in the config file

### 5.7 Number of iterations
Finally, we need to specify how many iterations should the training process be run for.
```python
max_batches=5200
```

For multi-class object detectors, the max_batches number is higher, i.e. we need to run for more number of batches(e.g. in yolov3-voc.cfg). 
For an n-classes object detector, it is advisable to run the training for at least 2000*n batches.

NOTE! This value is automatically calculated based in input classes.

## 6.Training YOLOv3
You can start training with following command:
```python
python train.py -m config/yolov3-tiny.cfg -d config/coco.data -e 100 --pretrained_weights weight/yolov3-tiny.weights -g 0
```
### 6.1 When do we stop the training?
train.py stops automatically when maximum number of batches is reached, but it is always advisable to monitor mAP, f1, 
recall and precision values and stop when desired values are reached.
## 7.Testing the model
You can test your model by typing:
```python
python test.py -m config/yolov3-tiny.cfg -d config/coco.data -w weight/yolov3-tiny.weights
```
## 8.Using autoDetect
You can use autoDetect to automatically detect new images on a folder
Define all needed parameters in a autodetect.cfg in config folder
```python
[autodetect]
#Directory
directory=C:/Users/UserX/Documents/images/camera_data/
# JSON path
json_path=
# Poll interval (seconds)
interval=60
#Use GPU
gpu = -1
# Model params
classes = config/custom.names
conf_thres = 0.35
nms_thres = 0.5
img_size = 640
model = config/yolov3-tiny.cfg
weights = weights/yolov3-tiny.weights
#SSH Params
# Connection parameters
host = hostname
# default SSH port
port = 22
username = username
password = password
```
Then start autodetect by typing:
```python
python autodetect.py
```

NOTE! SSH parts are not yet implemented! 
## 9. Using video detection
Start detecting objects from video by typing:
```python
python detectVideo.py -v <video_file_path> -cl <path_to_class_names> -c <path_to_config_file> -w <path_to_weights_file> -r <input resolution> -h <frame_hops>
```

## 10. Credits

### YOLOv3: An Incremental Improvement
_Joseph Redmon, Ali Farhadi_ <br>

**Abstract** <br>
We present some updates to YOLO! We made a bunch
of little design changes to make it better. We also trained
this new network that’s pretty swell. It’s a little bigger than
last time but more accurate. It’s still fast though, don’t
worry. At 320 × 320 YOLOv3 runs in 22 ms at 28.2 mAP,
as accurate as SSD but three times faster. When we look
at the old .5 IOU mAP detection metric YOLOv3 is quite
good. It achieves 57.9 AP50 in 51 ms on a Titan X, compared
to 57.5 AP50 in 198 ms by RetinaNet, similar performance
but 3.8× faster. As always, all the code is online at
https://pjreddie.com/yolo/.

[[Paper]](https://pjreddie.com/media/files/papers/YOLOv3.pdf) [[Project Webpage]](https://pjreddie.com/darknet/yolo/) [[Authors' Implementation]](https://github.com/pjreddie/darknet)

```
@article{yolov3,
  title={YOLOv3: An Incremental Improvement},
  author={Redmon, Joseph and Farhadi, Ali},
  journal = {arXiv},
  year={2018}
}
```

## 11. Other
### Sources
A minimal PyTorch implementation of YOLOv3.
- This code is based on following source codes:
- PyTorch-YOLOv3: https://github.com/eriklindernoren/PyTorch-YOLOv3
- YOLOv3 tutorial from scratch: https://github.com/ayooshkathuria/YOLO_v3_tutorial_from_scratch
- Pytorch - custom yolo training: https://github.com/cfotache/pytorch_custom_yolo_training
- Train your own yolo: https://github.com/AntonMu/TrainYourOwnYOLO
- Yolov3: https://github.com/ultralytics/yolov3
- Paper Yolo v4: https://arxiv.org/abs/2004.10934
- Original darknet source code:https://github.com/AlexeyAB/darknet
- More details: http://pjreddie.com/darknet/yolo/
- Training YOLOV3: Deep Learning Based Custom Object Detection: https://learnopencv.com/training-yolov3-deep-learning-based-custom-object-detector/
- Training Yolo for Object Detection in PyTorch with Your Custom Dataset — The Simple Way: https://towardsdatascience.com/training-yolo-for-object-detection-in-pytorch-with-your-custom-dataset-the-simple-way-1aa6f56cf7d9
- A PyTorch Extension for Learning Rate Warmup: https://github.com/Tony-Y/pytorch_warmup
- PyTorch implementation of YOLOv4: https://github.com/WongKinYiu/PyTorch_YOLOv4/blob/master/train.py
- Original code for this fork: 
# 12. What if's
### In case of Protobuf TypeError:
```
TypeError: Descriptors cannot not be created directly.
If this call came from a _pb2.py file, your generated code is out of date and must be regenerated with protoc >= 3.19.0.
If you cannot immediately regenerate your protos, some other possible workarounds are:
 1. Downgrade the protobuf package to 3.20.x or lower.
 2. Set PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python (but this will use pure-Python parsing and will be much slower).

More information: https://developers.google.com/protocol-buffers/docs/news/2022-05-06#python-updates

```
Downgrade protobuf: 
```
pip install protobuf==3.20.*
```
### OpenCV can't augment image: 608 x 608
https://github.com/Tianxiaomo/pytorch-YOLOv4/issues/427

### RuntimeError: shape '[1, 3, 29, 76, 76]' is invalid for input of size 1472880
https://github.com/Tianxiaomo/pytorch-YOLOv4/issues/138

# 9. Win install
### Installation of pycocotools

Install Microsoft VC >=14.0, remember to set correct PATH

https://github.com/philferriere/cocoapi

https://stackoverflow.com/questions/29846087/error-microsoft-visual-c-14-0-is-required-unable-to-find-vcvarsall-bat

https://stackoverflow.com/questions/29846087/error-microsoft-visual-c-14-0-is-required-unable-to-find-vcvarsall-bat/51087608#51087608
