#################################
# test.py
# Author: Juha-Matti Rouvinen
# Date: 2023-07-02
# Updated: 2024-01-25
# Version V2.1
##################################

from __future__ import division

import argparse
import datetime

import tqdm
from terminaltables import AsciiTable
from torch.autograd import Variable
from torch.utils.data import DataLoader, SequentialSampler, RandomSampler, SubsetRandomSampler, WeightedRandomSampler, \
    BatchSampler

from models import *
from utils.auc_roc import AUROC
from utils.confusion_matrix import ConfusionMatrix
from utils.datasets import ListDataset
from utils.datasets_v2 import LoadImagesAndLabels
from utils.parse_config import parse_data_config
from utils.plots import plot_images
from utils.torch_utils import time_synchronized
from utils.transforms import DEFAULT_TRANSFORMS
from utils.utils import load_classes, ap_per_class, get_batch_statistics, non_max_suppression, xywh2xyxy, \
    print_environment_info, get_class_weights


def evaluate_model_file(model_path, weights_path, img_path, class_names, batch_size, img_size,
                        n_cpu, iou_thres, conf_thres, nms_thres, verbose, device):
    """Evaluate model on validation dataset.

    :param model_path: Path to model definition file (.cfg)
    :type model_path: str
    :param weights_path: Path to weights or checkpoint file (.weights or .pth)
    :type weights_path: str
    :param img_path: Path to file containing all paths to validation images.
    :type img_path: str
    :param class_names: List of class names
    :type class_names: [str]
    :param batch_size: Size of each image batch, defaults to 8
    :type batch_size: int, optional
    :param img_size: Size of each image dimension for yolo, defaults to 416
    :type img_size: int, optional
    :param n_cpu: Number of cpu threads to use during batch generation, defaults to 8
    :type n_cpu: int, optional
    :param iou_thres: IOU threshold required to qualify as detected, defaults to 0.5
    :type iou_thres: float, optional
    :param conf_thres: Object confidence threshold, defaults to 0.5
    :type conf_thres: float, optional
    :param nms_thres: IOU threshold for non-maximum suppression, defaults to 0.5
    :type nms_thres: float, optional
    :param verbose: If True, prints stats of model, defaults to True
    :type verbose: bool, optional
    :return: Returns precision, recall, AP, f1, ap_class
    """
    if device.type == "cuda":
        gpu = 0
    else:
        gpu = -1
    dataloader = _create_validation_data_loader(
        img_path, batch_size, img_size, n_cpu)
    model = load_model(model_path, gpu, weights_path)

    metrics_output = _evaluate(
        model,
        dataloader,
        class_names,
        img_size,
        iou_thres,
        conf_thres,
        nms_thres,
        verbose,
        device)
    return metrics_output


def print_eval_stats(metrics_output, class_names, verbose):
    if metrics_output is not None:
        precision, recall, AP, f1, ap_class = metrics_output
        if verbose:
            # Prints class AP and mean AP
            ap_table = [["Index", "Class", "AP"]]
            for i, c in enumerate(ap_class):
                ap_table += [[c, class_names[c], "%.8f" % AP[i]]]
            print(AsciiTable(ap_table).table)
        print(f"---- mAP {AP.mean():.14f} ----")
    else:
        print("---- mAP not measured (no detections found by model) ----")


def clip_boxes(boxes, shape):
    # Clip boxes (xyxy) to image shape (height, width)
    if isinstance(boxes, torch.Tensor):  # faster individually
        boxes[..., 0].clamp_(0, shape[1])  # x1
        boxes[..., 1].clamp_(0, shape[0])  # y1
        boxes[..., 2].clamp_(0, shape[1])  # x2
        boxes[..., 3].clamp_(0, shape[0])  # y2
    else:  # np.array (faster grouped)
        boxes[..., [0, 2]] = boxes[..., [0, 2]].clip(0, shape[1])  # x1, x2
        boxes[..., [1, 3]] = boxes[..., [1, 3]].clip(0, shape[0])  # y1, y2


def scale_boxes(img1_shape, boxes, img0_shape, ratio_pad=None):
    # Rescale boxes (xyxy) from img1_shape to img0_shape
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    boxes[..., [0, 2]] -= pad[0]  # x padding
    boxes[..., [1, 3]] -= pad[1]  # y padding
    boxes[..., :4] /= gain
    clip_boxes(boxes, img0_shape)
    return boxes


def _evaluate(model, dataloader, class_names, img_log_path, epoch, draw,auc_roc, img_size, iou_thres, conf_thres, nms_thres,
              verbose, device, logger):
    """Evaluate model on validation dataset.

    :param model: Model to evaluate
    :type model: models.Darknet
    :param dataloader: Dataloader provides the batches of images with targets
    :type dataloader: DataLoader
    :param class_names: List of class names
    :type class_names: [str]
    :param img_size: Size of each image dimension for yolo
    :type img_size: int
    :param iou_thres: IOU threshold required to qualify as detected
    :type iou_thres: float
    :param conf_thres: Object confidence threshold
    :type conf_thres: float
    :param nms_thres: IOU threshold for non-maximum suppression
    :type nms_thres: float
    :param verbose: If True, prints stats of model
    :type verbose: bool
    :return: Returns precision, recall, AP, f1, ap_class
    :type: device: torch.device
    :return: is used to define gpu / cpu Tensor
    """
    # Performance improved version - 0.3.9
    model.eval()  # Set model to evaluation mode

    if not isinstance(model, torch.nn.Module):
        raise ValueError("model must be an instance of torch.nn.Module")

    #if not isinstance(dataloader, torch.utils.data.DataLoader):
    #    raise ValueError("dataloader must be an instance of torch.utils.data.DataLoader")

    if not device.type in ["cuda", "cpu"]:
        raise ValueError("Invalid device type")

    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is not available")

    if device.type == "cuda":
        Tensor = torch.cuda.FloatTensor
    else:
        Tensor = torch.FloatTensor

    labels = []
    sample_metrics = []  # List of tuples (TP, confs, pred)

    confusion_matrix = ConfusionMatrix(nc=len(class_names))


    p, r, f1, mp, mr, map50, map, t0, t1 = 0., 0., 0., 0., 0., 0., 0., 0., 0.
    s = ('%20s' + '%12s' * 6) % ('Class', 'Images', 'Targets', 'P', 'R', 'mAP@.5', 'mAP@.5:.95')
    if draw or auc_roc:
        if device.type == "cuda":
            eval_plot_outputs = None
            eval_plot_targets = None
        else:
            eval_plot_outputs = torch.tensor(data='')
            eval_plot_targets = torch.tensor(data='')

    for _, imgs, targets in tqdm.tqdm(dataloader, desc="Validating",colour='green'):
    #for batch_i, (img, targets, paths, shapes) in enumerate(tqdm(dataloader, desc=s)):
    #for imgs, targets in tqdm.tqdm(dataloader, desc="Validating"):
        # Extract labels
        labels += targets[:, 1].tolist()
        # Rescale target
        targets[:, 2:] = xywh2xyxy(targets[:, 2:])
        targets[:, 2:] *= img_size
        imgs = Variable(imgs.type(Tensor), requires_grad=False)
        with torch.no_grad():
            t = time_synchronized()
            outputs = model(imgs)
            t0 += time_synchronized() - t
            t = time_synchronized()
            if draw or auc_roc:
                if eval_plot_outputs == None:
                    eval_plot_outputs = outputs
                    eval_plot_targets = targets
                else:
                    # Append eval tensors
                    torch.cat((eval_plot_outputs, outputs))
                    torch.cat((eval_plot_targets, targets))



            outputs = non_max_suppression(
                outputs, conf_thres=conf_thres, iou_thres=nms_thres
            )
            t1 += time_synchronized() - t

        sample_metrics.extend(get_batch_statistics(outputs, targets, iou_threshold=iou_thres))

    if draw or auc_roc:
        eval_plot_outputs = non_max_suppression(
            eval_plot_outputs, conf_thres=conf_thres, iou_thres=nms_thres
        )
    if draw:
        # Confusion matrix
        confusion_matrix.generate_batch_data(eval_plot_outputs, eval_plot_targets)
        confusion_matrix.plot(True, img_log_path, class_names,epoch,logger)

    if auc_roc:
        aucroc = AUROC(nc=len(class_names), conf=conf_thres, iou_thres=iou_thres)
        names = model.names if hasattr(model, 'names') else model.module.names  # get class names
        if isinstance(names, (list, tuple)):  # old format
            names = dict(enumerate(names))
        # auc roc
        aucroc.generate_batch_data(eval_plot_outputs, eval_plot_targets)
        # Compute AUC
        auc_scores, fpr_, tpr_ = aucroc.out()
        mauc = auc_scores.mean()
        if float(mauc) > 0.0:
            new_name = ['AUC/' + i for i in names.values()]
            auc_scores_name = dict(zip(new_name, auc_scores))
            auc_scores_name['AUC/mAUC'] = mauc
            aucroc.plot_auroc_curve(fpr_, tpr_, auc_scores, img_log_path, names, epoch,logger)
            aucroc.plot_polar_chart(auc_scores,img_log_path,names)
        else:
            print(f"- ❎ - No detections in validation set -> skipping AOC ROC plotting ----")

    #if len(sample_metrics) == 0:  # No detections over whole validation set.
    #    print("---- No detections over whole validation set ----")
    #    return None

    # Compute statistics
    '''
    stats = [np.concatenate(x, 0) for x in zip(*sample_metrics)]  # to numpy
    if len(stats) and stats[0].any():
        p, r, ap, f1, ap_class = ap_per_class(*stats, plot=plots, fname='./logs/precision-recall_curve.png')
        p, r, ap50, ap = p[:, 0], r[:, 0], ap[:, 0], ap.mean(1)  # [P, R, AP@0.5, AP@0.5:0.95]
        mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()
        nt = np.bincount(stats[3].astype(np.int64), minlength=num_classes)  # number of targets per class
    else:
        nt = torch.zeros(1)
    '''

    # Concatenate sample statistics
    true_positives, pred_scores, pred_labels = [
        np.concatenate(x, 0) for x in list(zip(*sample_metrics))]
    metrics_output = ap_per_class(true_positives, pred_scores, pred_labels, labels)
    print_eval_stats(metrics_output, class_names, verbose)
    # Print speeds
    # t = tuple(x / seen * 1E3 for x in (t0, t1, t0 + t1)) + (img_size, img_size, batch_size)  # tuple
    # print('Speed: %.1f/%.1f/%.1f ms inference/NMS/total per %gx%g image at batch-size %g' % t)
    return metrics_output, outputs, targets

def _create_validation_data_loader_v2(img_path, batch_size, img_size, n_cpu,sampler,class_names,img_log_path):
    """
    Creates a DataLoader for validation.

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
    #if sampler == 0:
    dataset = ListDataset(img_path, img_size=img_size, multiscale=False, transform=DEFAULT_TRANSFORMS)
    shuffle = True
    #else:
    #    dataset = LoadImagesAndLabels(img_path, img_size, batch_size,
    #                              augment=False,  # augment images
    #                              )
    #    shuffle = False
    # batch_size = min(batch_size, len(dataset))
    # [0 = None, 1 = SequentialSampler, 2 = RandomSampler, 3 = SubsetRandomSampler, 4 = WeightedRandomSampler, 5 = BatchSampler]
    #set_suffle = True
    if sampler == 1:
        sampler = SequentialSampler(
            data_source=dataset
        )
        #set_suffle = False

    elif sampler == 2:
        # class_weights_all = get_class_weights(dataset, class_names, "orig",img_log_path)
        sampler = RandomSampler(
            data_source=dataset,
            num_samples=batch_size,
        )
        #set_suffle = False

    elif sampler == 3:
        sampler = SubsetRandomSampler(
            indices=[10, 20, 30, 40, 50]

        )
        #set_suffle = False

    elif sampler == 4:
        # https://towardsdatascience.com/pytorch-basics-sampling-samplers-2a0f29f0bf2a
        # https://towardsdatascience.com / demystifying - pytorchs - weightedrandomsampler - by - example - a68aceccb452
        # class_weights_all = get_class_weights(dataset, class_names, "orig", img_log_path)
        class_weights_orig, class_count = get_class_weights(dataset, class_names, "evaluate", img_log_path)
        sampler = WeightedRandomSampler(
            num_samples=batch_size * 2,
            # Adjusting this parameter to double the size of our original dataset,
            # we can see that more of our images are seen over the course of an epoch.
            weights=class_weights_orig,
            replacement=True
        )
        #set_suffle = False

    elif sampler == 5:
        sampler = BatchSampler(
            SequentialSampler(range(10)),
            batch_size=3,
            drop_last=True
        )
        #set_suffle = False

    else:
        sampler = None

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=n_cpu,
        sampler=sampler,
        pin_memory=True,
        collate_fn=dataset.collate_fn)
    return dataloader

def _create_validation_data_loader_v1(img_path, batch_size, img_size, n_cpu):
    """
    Creates a DataLoader for validation.

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
    dataset = ListDataset(img_path, img_size=img_size, multiscale=False, transform=DEFAULT_TRANSFORMS)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=n_cpu,
        pin_memory=True,
        collate_fn=dataset.collate_fn)
    return dataloader


def run():
    date = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    ver = "0.3.15"
    # print_environment_info()
    print_environment_info(ver, "logs/" + date + "_log" + ".txt")
    parser = argparse.ArgumentParser(description="Evaluate validation data.")
    parser.add_argument("-m", "--model", type=str, default="config/yolov3.cfg",
                        help="Path to model definition file (.cfg)")
    parser.add_argument("-w", "--weights", type=str, default="weights/yolov3.weights",
                        help="Path to weights or checkpoint file (.weights or .pth)")
    parser.add_argument("-d", "--data", type=str, default="config/coco.data", help="Path to data config file (.data)")
    parser.add_argument("-b", "--batch_size", type=int, default=8, help="Size of each image batch")
    parser.add_argument("-v", "--verbose", action='store_true', help="Makes the validation more verbose")
    parser.add_argument("--img_size", type=int, default=416, help="Size of each image dimension for yolo")
    parser.add_argument("--n_cpu", type=int, default=4, help="Number of cpu threads to use during batch generation")
    parser.add_argument("--iou_thres", type=float, default=0.5, help="IOU threshold required to qualify as detected")
    parser.add_argument("--conf_thres", type=float, default=0.01, help="Object confidence threshold")
    parser.add_argument("--nms_thres", type=float, default=0.4, help="IOU threshold for non-maximum suppression")
    args = parser.parse_args()
    print(f"Command line arguments: {args}")

    # Load configuration from data file
    data_config = parse_data_config(args.data)
    # Path to file containing all images for validation
    valid_path = data_config["valid"]
    class_names = load_classes(data_config["names"])  # List of class names
    verbose = True
    # GPU determination
    if torch.cuda.is_available() is True:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f'Using cuda device - {device}')
    precision, recall, AP, f1, ap_class = evaluate_model_file(
        args.model,
        args.weights,
        valid_path,
        class_names,
        batch_size=args.batch_size,
        img_size=args.img_size,
        n_cpu=args.n_cpu,
        iou_thres=args.iou_thres,
        conf_thres=args.conf_thres,
        nms_thres=args.nms_thres,
        verbose=verbose,
        device=device
    )


if __name__ == "__main__":
    run()
