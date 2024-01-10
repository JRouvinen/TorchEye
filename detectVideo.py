# YOLO v3 Video Detection Module

# This code implements an object detection module using YOLO v3. It takes a video as input and detects objects in each frame of the video. The detected objects are then annotated with bounding boxes and class labels. The annotated frames are displayed in real-time.

# The code first imports the required libraries and defines some utility functions. It then parses the command-line arguments to get the input video file, confidence threshold, NMS threshold, YOLO configuration file, YOLO weights file, and input resolution. It also loads the class labels and assigns colors to each class.

# The YOLO model is loaded and the video capture is initialized. The code then processes each frame of the video. For every 'frame_hop' frames, the frame is preprocessed and input to the YOLO model for object detection. The detected objects are filtered based on the confidence threshold and NMS threshold. The filtered objects are then annotated with bounding boxes and class labels. The annotated frame is displayed in real-time.

# The code calculates the frames per second (FPS) and displays it in the console. The video processing continues until the end of the video or the 'q' key is pressed.

# The code is written in Python and requires the following libraries: torch, numpy, opencv-python, pandas, argparse.

from __future__ import division
import time
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import cv2

from utils.parse_config import parse_hyp_config
from utils.video_utils import *
from models import Darknet, load_model
from utils.preprocess import prep_image, inp_to_image, letterbox_image
import pandas as pd
import random
import pickle as pkl
import argparse
from detect import detect_image, draw_and_save_return_image

# YOLO configuration

ver = "0.2"
colors = [
        (0, 0, 255),  # Red
        (0, 255, 0),  # Green
        (255, 0, 0),  # Blue
        (255, 255, 0),  # Yellow
        (255, 0, 255),  # Magenta
        (0, 255, 255),  # Cyan
        (255, 128, 0),  # Orange
        (128, 0, 255),  # Purple
        (0, 128, 255),  # Light Blue
        (128, 255, 0),  # Lime Green
        (255, 128, 128),  # Light Red
        (128, 128, 255),  # Light Blue
        (0, 128, 128),  # Teal
        (128, 0, 0),  # Maroon
        (0, 0, 128),  # Navy
        (128, 0, 128),  # Purple
        (0, 128, 0),  # Dark Green
        (128, 128, 0),  # Olive
        (128, 255, 128),  # Pale Green
        (255, 128, 255),  # Pink
        (128, 128, 128),  # Gray
        (255, 255, 255),  # White
        (0, 0, 0),  # Black
        (255, 0, 128),  # Hot Pink
        (128, 0, 64),  # Burgundy
        (192, 192, 192),  # Silver
        (255, 165, 0),  # Orange
        (255, 215, 0),  # Gold
        (0, 255, 128),  # Spring Green
        (0, 128, 64),  # Forest Green
        (0, 128, 128),  # Dark Cyan
        (0, 64, 128),  # Steel Blue
        (0, 0, 160),  # Navy Blue
        (0, 255, 255),  # Light Cyan
        (0, 128, 192),  # Cerulean
        (0, 128, 255),  # Sky Blue
        (0, 0, 255),  # Bright Blue
        (255, 192, 203),  # Pink
        (255, 0, 0),  # Bright Red
        (255, 165, 0),  # Orange
        (255, 255, 0),  # Bright Yellow
        (0, 255, 0),  # Bright Green
        (0, 255, 255),  # Bright Cyan
        (0, 0, 128),  # Dark Blue
        (128, 0, 128),  # Dark Purple
        (128, 128, 0),  # Olive
        (128, 128, 128),  # Dark Gray
        (255, 255, 255),  # White
        (0, 0, 0)  # Black
    ]

# Function to get test input

def get_test_input(input_dim, CUDA):
    img = cv2.imread("./config/test_image.jpg")
    img = cv2.resize(img, (input_dim, input_dim))
    img_ = img[:, :, ::-1].transpose((2, 0, 1))
    img_ = img_[np.newaxis, :, :, :] / 255.0
    img_ = torch.from_numpy(img_).float()
    img_ = Variable(img_)

    if CUDA:
        img_ = img_.cuda()

    return img_

# Function to prepare image for inputting to the neural network
def prep_image(img, inp_dim):
    """
    Prepare image for inputting to the neural network.

    Returns a Variable
    """

    orig_im = img
    dim = orig_im.shape[1], orig_im.shape[0]
    img = (letterbox_image(orig_im, (inp_dim, inp_dim)))
    img_ = img[:, :, ::-1].transpose((2, 0, 1)).copy()
    img_ = torch.from_numpy(img_).float().div(255.0).unsqueeze(0)
    return img_, orig_im, dim

# Function to assign colors to classes
def assign_colors_to_classes(num_classes, color_list):
    class_colors = {}
    for i in range(num_classes):
        class_index = i % len(color_list)
        class_colors[i] = color_list[class_index]
    return class_colors

# Function to parse command-line arguments
def arg_parse():
    """
    Parse arguments to the detect module

    """

    parser = argparse.ArgumentParser(description='YOLO v3 Video Detection Module')

    parser.add_argument("-v", "--video", dest='video', help=
    "Video to run detection upon",
                        default="video.avi", type=str)
    parser.add_argument("-cl", "--classes", dest="classes", help="Classes file", default="data/coco.names")
    parser.add_argument("-conf", "--confidence", dest="confidence", help="Object Confidence to filter predictions",
                        default=0.3)
    parser.add_argument("-nms", "--nms_thresh", dest="nms_thresh", help="NMS Threshhold", default=0.3)
    parser.add_argument("-c", "--cfg", dest='cfgfile', help=
    "Config file",
                        default="cfg/yolov3.cfg", type=str)
    parser.add_argument("--hyp", type=str, default="config/hyp.cfg",
                        help="Path to hyperparameters config file (.cfg)")
    parser.add_argument("-w", "--weights", dest='weightsfile', help=
    "weightsfile",
                        default="yolov3.weights", type=str)
    parser.add_argument("-r", "--reso", dest='reso', help=
    "Input resolution of the network. Increase to increase accuracy. Decrease to increase speed",
                        default="640", type=str)
    parser.add_argument("-fh", "--frame_hop", dest='frame_hop', help=
    "Determine how many frames should be discarded. Increase to increase speed.",
                        default=24, type=int)
    return parser.parse_args()


if __name__ == '__main__':
    # Parse command-line arguments
    args = arg_parse()
    # Parse hyperparameters
    hyp_config = parse_hyp_config(args.hyp)
    # Set confidence and NMS thresholds
    confidence = float(args.confidence)
    nms_thesh = float(args.nms_thresh)
    # Load class labels
    classes = load_classes(args.classes)
    num_classes = len(classes)
    # Check if CUDA is avalable
    CUDA = torch.cuda.is_available()
    # Create color mapping
    class_colors = assign_colors_to_classes(num_classes, colors)
    # CUDA = torch.cuda.is_available()

    bbox_attrs = 5 + num_classes
    # Initialize YOLO model
    print("Loading network.....")
    #model = Darknet(args.cfgfile,hyp_config)
    #model.load_darknet_weights(args.weightsfile)
    model = load_model(args.cfgfile, hyp_config, -1, args.weightsfile)
    model.eval()  # Set model to evaluation mode
    print("Network successfully loaded")

    model.hyperparams["height"] = args.reso
    inp_dim = int(model.hyperparams["height"])
    assert inp_dim % 32 == 0
    assert inp_dim > 32

    # Set CUDA if CUDA is available
    if CUDA:
        model.cuda()

    model(get_test_input(inp_dim, CUDA))

    # Initialize video capture
    videofile = args.video

    cap = cv2.VideoCapture(videofile)

    assert cap.isOpened(), 'Cannot capture source'

    # Initialize variables
    frames = 0
    frame_hop = int(args.frame_hop)
    start = time.time()
    # Process each frame of the video
    while cap.isOpened():
        if frames % frame_hop == 0:

            ret, frame = cap.read()
            if ret:
                # Preprocess frame
                img, orig_im, dim = prep_image(frame, inp_dim)

                im_dim = torch.FloatTensor(dim).repeat(1, 2)
                cv2.imwrite("video_to_annotation.jpg", orig_im)
                # Detect objects in the frame
                img_detections, imgs = detect_image(model, img, int(args.reso), confidence,
                                                    nms_thesh)  # model, image, img_size=416, conf_thres=0.5, nms_thres=0.5
                # Annotate frame with detections
                img_with_detection = draw_and_save_return_image(orig_im, img_detections[0], int(args.reso), classes, class_colors)
                cv2.imwrite("video_annotation.jpg", img_with_detection)
                # Display annotated frame
                if type(img_with_detection) == int:
                    frames += 1
                    print("FPS of the video is {:5.2f}".format(frames / (time.time() - start)))
                    cv2.imshow("frame", img_with_detection)
                    key = cv2.waitKey(1)
                    if key & 0xFF == ord('q'):
                        break
                    continue

                cv2.imshow("frame", img_with_detection)
                key = cv2.waitKey(1)
                if key & 0xFF == ord('q'):
                    break
                frames += 1
                print("FPS of the video is {:5.2f}".format(frames / (time.time() - start)))


            else:
                break

        else:
            frames += 1
