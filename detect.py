#! /usr/bin/env python3
#################################
# detect.py
# Author: Juha-Matti Rouvinen
# Date: 2023-07-02
# Updated: 2024-01-16
# Version V2
##################################
from __future__ import division

import os
import argparse
import datetime
import time

import cv2
import tqdm
import random
import numpy as np

from PIL import Image

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable

from models import load_model
from utils.parse_config import parse_hyp_config
from utils.utils import load_classes, rescale_boxes, non_max_suppression, print_environment_info
from utils.datasets import ImageFolder, ListDataset, ImageList
from utils.transforms import Resize, DEFAULT_TRANSFORMS

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import NullLocator

from utils.writer import log_file_writer
from profilehooks import profile


def detect_directory(model_path, weights_path, img_path, classes, output_path, gpu, date, hyp,
                     batch_size=8, img_size=416, n_cpu=8, conf_thres=0.5, nms_thres=0.5, draw=0):
    """Detects objects on all images in specified directory and saves output images with drawn detections.

    :param model_path: Path to model definition file (.cfg)
    :type model_path: str
    :param weights_path: Path to weights or checkpoint file (.weights or .pth)
    :type weights_path: str
    :param img_path: Path to directory with images to inference
    :type img_path: str
    :param classes: List of class names
    :type classes: [str]
    :param output_path: Path to output directory
    :type output_path: str
    :param batch_size: Size of each image batch, defaults to 8
    :type batch_size: int, optional
    :param img_size: Size of each image dimension for yolo, defaults to 416
    :type img_size: int, optional
    :param n_cpu: Number of cpu threads to use during batch generation, defaults to 8
    :type n_cpu: int, optional
    :param conf_thres: Object confidence threshold, defaults to 0.5
    :type conf_thres: float, optional
    :param nms_thres: IOU threshold for non-maximum suppression, defaults to 0.5
    :type nms_thres: float, optional
    """
    # draw=0
    dataloader = _create_data_loader(img_path, batch_size, img_size, n_cpu)
    model = load_model(model_path, hyp, gpu, weights_path)
    img_detections, imgs = detect(
        model,
        dataloader,
        output_path,
        conf_thres,
        nms_thres,
        gpu)
    # if draw != 0:
    _draw_and_save_output_images(
        img_detections, imgs, img_size, output_path, classes, date, draw)

    print(f"---- Detections were saved to: '{output_path}' ----")


def detect_image(model, image, img_size=416, conf_thres=0.35, nms_thres=0.4):
    """Inferences one image with model.

    :param model: Model for inference
    :type model: models.Darknet
    :param image: Image to inference
    :type image: nd.array
    :param img_size: Size of each image dimension for yolo, defaults to 416
    :type img_size: int, optional
    :param conf_thres: Object confidence threshold, defaults to 0.5
    :type conf_thres: float, optional
    :param nms_thres: IOU threshold for non-maximum suppression, defaults to 0.5
    :type nms_thres: float, optional
    :return: Detections on image with each detection in the format: [x1, y1, x2, y2, confidence, class]
    :rtype: nd.array
    """
    # model.eval()  # Set model to evaluation mode

    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
    img_detections = []  # Stores detections for each image index
    imgs = []  # Stores image paths

    input_imgs = Variable(image.type(Tensor))

    # Get detections
    with torch.no_grad():
        detections = model(input_imgs)
        detections = non_max_suppression(detections, conf_thres, nms_thres)

    # Store image and detections
    img_detections.extend(detections)
    # imgs.extend(img_paths)
    return img_detections, imgs

# Detect images is depricated in version 0.2.2
def detect_images(model, images, batch_size, img_size=640, conf_thres=0.5, nms_thres=0.5, gpu=-1):
    """Inferences one image with model.
    :param model: Model for inference
    :type model: models.Darknet
    :param image: Image to inference
    :type image: nd.array
    :param img_size: Size of each image dimension for yolo, defaults to 416
    :type img_size: int, optional
    :param conf_thres: Object confidence threshold, defaults to 0.5
    :type conf_thres: float, optional
    :param nms_thres: IOU threshold for non-maximum suppression, defaults to 0.5
    :type nms_thres: float, optional
    :return: Detections on image with each detection in the format: [x1, y1, x2, y2, confidence, class]
    :rtype: nd.array
    """
    print('input images:', images)
    # batch_size = 16
    n_cpu = 4
    #model.eval()  # Set model to evaluation mode
    dataloader = _create_dataloader_from_list(images, batch_size, img_size, n_cpu)
    img_detections, imgs = detect(model, dataloader, None, 0.3, 0.5, gpu)
    return img_detections, imgs

    '''
    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
    img_detections = []  # Stores detections for each image index
    imgs = [] # Stores image paths
    for image in dataloader:
        input_imgs = Variable(image.type(Tensor))

        # Get detections
        with torch.no_grad():
            detections = model(input_imgs)
            detections = non_max_suppression(detections, conf_thres, nms_thres)

        # Store image and detections
            # Store image and detections
            img_detections.extend(detections)
            imgs.extend(images)
    #imgs.extend(img_paths)
    return img_detections, imgs
    '''


def detect(model, dataloader, output_path, conf_thres, nms_thres, gpu):
    """Inferences images with model.

    :param model: Model for inference
    :type model: models.Darknet
    :param dataloader: Dataloader provides the batches of images to inference
    :type dataloader: DataLoader
    :param output_path: Path to output directory
    :type output_path: str
    :param conf_thres: Object confidence threshold, defaults to 0.5
    :type conf_thres: float, optional
    :param nms_thres: IOU threshold for non-maximum suppression, defaults to 0.5
    :type nms_thres: float, optional
    :return: List of detections. The coordinates are given for the padded image that is provided by the dataloader.
        Use `utils.rescale_boxes` to transform them into the desired input image coordinate system before its transformed by the dataloader),
        List of input image paths
    :rtype: [Tensor], [str]
    """
    if output_path != None:
        # Create output directory, if missing
        os.makedirs(output_path, exist_ok=True)

    model.eval()  # Set model to evaluation mode

    # Old implementation -> caused RuntimeError: Input type (torch.cuda.FloatTensor) and
    # weight type (torch.FloatTensor) should be the same if user wanted to use cpu in machine with cuda gpu
    # Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
    if gpu != -1 and torch.cuda.is_available():
        Tensor = torch.cuda.FloatTensor
    else:
        Tensor = torch.FloatTensor

    img_detections = []  # Stores detections for each image index
    imgs = []  # Stores image paths

    for (img_paths, input_imgs) in tqdm.tqdm(dataloader, desc="Detecting"):
        # Configure input
        input_imgs = Variable(input_imgs.type(Tensor))

        # Get detections
        with torch.no_grad():
            detections = model(input_imgs)
            detections = non_max_suppression(detections, conf_thres, nms_thres)

        # Store image and detections
        img_detections.extend(detections)
        imgs.extend(img_paths)
    return img_detections, imgs


def _draw_and_save_output_images(img_detections, imgs, img_size, output_path, classes, date, draw):
    """Draws detections in output images and stores them.

    :param img_detections: List of detections
    :type img_detections: [Tensor]
    :param imgs: List of paths to image files
    :type imgs: [str]
    :param img_size: Size of each image dimension for yolo
    :type img_size: int
    :param output_path: Path of output directory
    :type output_path: str
    :param classes: List of class names
    :type classes: [str]
    """

    # Iterate through images and save plot of detections
    for (image_path, detections) in zip(imgs, img_detections):
        if draw != 0:
            print(f"Image {image_path}:")
        log_file_writer(f"Image {image_path}:", "output/" + date + "_detect" + ".txt")
        _draw_and_save_output_image(
            image_path, detections, img_size, output_path, classes, date, draw)


def _draw_and_save_output_image(image_path, detections, img_size, output_path, classes, date, draw):
    """Draws detections in output image and stores this.

    :param image_path: Path to input image
    :type image_path: str
    :param detections: List of detections on image
    :type detections: [Tensor]
    :param img_size: Size of each image dimension for yolo
    :type img_size: int
    :param output_path: Path of output directory
    :type output_path: str
    :param classes: List of class names
    :type classes: [str]
    """
    # create just log file
    if draw == 0:
        img = np.array(Image.open(image_path))
        # Rescale boxes to original image
        detections = rescale_boxes(detections, img_size, img.shape[:2])
        unique_labels = detections[:, -1].cpu().unique()
        n_cls_preds = len(unique_labels)
        for x1, y1, x2, y2, conf, cls_pred in detections:
            # print(f"\t+ Label: {classes[int(cls_pred)]} | Confidence: {conf.item():0.4f}")
            log_file_writer(f"\t+ Label: {classes[int(cls_pred)]} | Confidence: {conf.item():0.4f}",
                            "output/" + date + "_detect" + ".txt")
    else:
        # Create plot
        img = np.array(Image.open(image_path))
        plt.figure()
        fig, ax = plt.subplots(1)
        ax.imshow(img)
        # Rescale boxes to original image
        detections = rescale_boxes(detections, img_size, img.shape[:2])
        unique_labels = detections[:, -1].cpu().unique()
        n_cls_preds = len(unique_labels)
        # Bounding-box colors
        cmap = plt.get_cmap("tab20b")
        colors = [cmap(i) for i in np.linspace(0, 1, n_cls_preds)]
        bbox_colors = random.sample(colors, n_cls_preds)
        for x1, y1, x2, y2, conf, cls_pred in detections:
            print(f"\t+ Label: {classes[int(cls_pred)]} | Confidence: {conf.item():0.4f}")
            log_file_writer(f"\t+ Label: {classes[int(cls_pred)]} | Confidence: {conf.item():0.4f}",
                            "output/" + date + "_detect" + ".txt")

            box_w = x2 - x1
            box_h = y2 - y1

            color = bbox_colors[int(np.where(unique_labels == int(cls_pred))[0][0])]
            # Create a Rectangle patch
            bbox = patches.Rectangle((x1, y1), box_w, box_h, linewidth=2, edgecolor=color, facecolor="none")
            # Add the bbox to the plot
            ax.add_patch(bbox)
            # Add label
            plt.text(
                x1,
                y1,
                s=f"{classes[int(cls_pred)]}: {conf:.2f}",
                color="white",
                verticalalignment="top",
                bbox={"color": color, "pad": 0})

        # Save generated image with detections
        plt.axis("off")
        plt.gca().xaxis.set_major_locator(NullLocator())
        plt.gca().yaxis.set_major_locator(NullLocator())
        filename = os.path.basename(image_path).split(".")[0]
        output_path = os.path.join(output_path, f"{filename}.png")
        plt.savefig(output_path, bbox_inches="tight", pad_inches=0.0)
        plt.close()


def draw_and_save_return_image(image, detections, img_size, classes, class_colors):
    """Draws detections in output image and stores this.

        :param image_path: Path to input image
        :type image_path: str
        :param detections: List of detections on image
        :type detections: [Tensor]
        :param img_size: Size of each image dimension for yolo
        :type img_size: int
        :param output_path: Path of output directory
        :type output_path: str
        :param classes: List of class names
        :type classes: [str]
        """

    # Create plot
    img = np.array(image)

    # Rescale boxes to original image
    detections = rescale_boxes(detections, img_size, img.shape[:2])
    unique_labels = detections[:, -1].cpu().unique()
    n_cls_preds = len(unique_labels)

    for x1, y1, x2, y2, conf, cls_pred in detections:
        box_w = x2 - x1
        box_h = y2 - y1

        color = class_colors[int(cls_pred)]

        x1_loc = int(x1.item())
        y1_loc = int(y1.item())
        x2_loc = int(x2.item())
        y2_loc = int(y2.item())
        # cv2.parts
        cv2.rectangle(image, (x1_loc, y1_loc), (x2_loc, y2_loc), color, 1)
        t_size = cv2.getTextSize(classes[int(cls_pred)], cv2.FONT_HERSHEY_PLAIN, 1, 1)[0]
        c2 = x1_loc + t_size[0] + 40, y1_loc + t_size[1] + 4
        cv2.rectangle(image, (x1_loc, y1_loc), c2, color, -1)
        cv2.putText(image, classes[int(cls_pred)] + f',{round(float(conf), 2)}', (x1_loc, y1_loc + t_size[1] + 4),
                    cv2.FONT_HERSHEY_PLAIN, 1, [225, 255, 255], 1);

    return image


def _create_dataloader_from_list(img_list, batch_size, img_size, n_cpu):
    dataset = ImageList(
        img_list,
        transform=transforms.Compose([DEFAULT_TRANSFORMS, Resize(img_size)]))

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=n_cpu,
        pin_memory=True)
    return dataloader


def _create_data_loader_list(list_path, batch_size, img_size, n_cpu):
    dataset = ListDataset(
        list_path,
        img_size,
        multiscale=True,
        transform=None

    )
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=n_cpu,
        pin_memory=True)
    return dataloader


def _create_data_loader(img_path, batch_size, img_size, n_cpu):
    """Creates a DataLoader for inferencing.

    :param img_path: Path to file containing all paths to validation images.
    :type img_path: str
    :param batch_size: Size of each image batch
    :type batch_size: int
    :param img_size: Size of each image dimension for yolo
    :type img_size: int
    :param n_cpu: Number of cpu threads to use during batch generation
    :type n_cpu: int
    :return: Returns DataLoader
    :rtype: DataLoader
    """
    dataset = ImageFolder(
        img_path,
        transform=transforms.Compose([DEFAULT_TRANSFORMS, Resize(img_size)]))
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=n_cpu,
        pin_memory=True)
    return dataloader


@profile(filename='./logs/profiles/detect.prof', stdout=False)
def run():
    date = datetime.datetime.now().strftime("%Y_%m_%d__%H_%M_%S")
    ver = "0.2.2"
    print_environment_info(ver, "output/" + date + "_detect" + ".txt")
    parser = argparse.ArgumentParser(description="Detect objects on images.")
    parser.add_argument("-m", "--model", type=str, default="config/yolov3.cfg",
                        help="Path to model definition file (.cfg)")
    parser.add_argument("-w", "--weights", type=str, default="weights/yolov3.weights",
                        help="Path to weights or checkpoint file (.weights or .pth)")
    parser.add_argument("-i", "--images", type=str, default="data/samples",
                        help="Path to directory with images to inference")
    parser.add_argument("-c", "--classes", type=str, default="data/coco.names",
                        help="Path to classes label file (.names)")
    parser.add_argument("--hyp", type=str, default="config/hyp.cfg",
                        help="Path to hyperparameters config file (.cfg)")
    parser.add_argument("-o", "--output", type=str, default="output", help="Path to output directory")
    parser.add_argument("-b", "--batch_size", type=int, default=4, help="Size of each image batch")
    parser.add_argument("-d", "--draw", type=int, default=0, help="Draw detection boxes into images")
    parser.add_argument("--img_size", type=int, default=832, help="Size of each image dimension for yolo")
    parser.add_argument("--n_cpu", type=int, default=4, help="Number of cpu threads to use during batch generation")
    parser.add_argument("--conf_thres", type=float, default=0.35, help="Object confidence threshold")
    parser.add_argument("--nms_thres", type=float, default=0.5, help="IOU threshold for non-maximum suppression")
    parser.add_argument("-g", "--gpu", type=int, default=-1, help="GPU to use")

    args = parser.parse_args()
    print(f"Command line arguments: {args}")
    # Create_new detect_file
    f = open("output/" + date + "_detect" + ".txt", "w")
    f.close()
    log_file_writer(f"Command line arguments: {args}", "output/" + date + "_detect" + ".txt")

    # Extract class names from file
    classes = load_classes(args.classes)  # List of class names
    # Get hyperparameters configuration
    hyp_config = parse_hyp_config(args.hyp)
    detect_directory(
        args.model,
        args.weights,
        args.images,
        classes,
        args.output,
        args.gpu,
        date,
        hyp_config,
        batch_size=args.batch_size,
        img_size=args.img_size,
        n_cpu=args.n_cpu,
        conf_thres=args.conf_thres,
        nms_thres=args.nms_thres,
        draw=args.draw
    )


if __name__ == '__main__':
    # Start timer
    start_time = time.time()

    # Code to be timed
    # ...
    run()

    # End timer
    end_time = time.time()

    # Calculate elapsed time
    elapsed_time = end_time - start_time
    print("Elapsed time: ", elapsed_time)
