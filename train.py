#! /usr/bin/env python3
#################################
# train.py
# Author: Juha-Matti Rouvinen
# Date: 2023-07-01
##################################
'''
The  train.py  script is used to train the YOLO (You Only Look Once)
object detection model.
It takes command line arguments to specify the model definition file,
data configuration file, number of epochs,
verbosity level, GPU usage, and other parameters.

The script starts by creating necessary folders for logs,
checkpoints, and output.
It then parses the command line arguments and loads the model,
data configuration, and class names.
The model is loaded with the specified pretrained weights if provided.

Next, the script creates a DataLoader for training and validation data.
It also creates an optimizer for the model based on the specified
optimizer in the model's hyperparameters.
The learning rate is adjusted based on the number
of warmup iterations or the scheduler.

The script then trains the model for the specified number of epochs.
In each epoch, it iterates over the training data,
performs forward and backward passes, and updates the model's parameters.
It logs the training progress and saves checkpoints at specified intervals.

After each epoch, the script evaluates the model on the validation set.
It calculates precision, recall, average precision (AP), and F1 score.
It also calculates a fitness score based on these metrics.
The best model based on the fitness score is saved as
the best checkpoint.

The script logs the training and evaluation progress to TensorBoard and ClearML, if enabled.
It also saves training and evaluation metrics to CSV files for further analysis.

Finally, the script prints the execution time and provides a command to
monitor training progress with TensorBoard.

To run the script, you need to provide the necessary command line arguments,
such as the model definition file,data configuration file, and pretrained weights.
For example:

python train.py -m config/yolov3.cfg -d config/coco.data -e 300 -v
--pretrained_weights weights/yolov3.weights --checkpoint_interval 5 --evaluation_interval 5

This will train the YOLO model for 300 epochs using the COCO dataset, with verbosity and save checkpoints and
perform evaluations every 5 epochs. The pretrained weights from  weights/yolov3.weights  will be used to initialize the model.

'''

from __future__ import division

import math
import os
import argparse
import datetime
import shutil
import sys
import time
import traceback
import configparser
from test import _evaluate, _create_validation_data_loader
import tqdm
import torch
from torch.optim.lr_scheduler import ConstantLR, ExponentialLR
from torch.utils.data import DataLoader
import torch.optim as optim
import numpy as np
from torchsummary import summary
from tensorboard import program
from terminaltables import AsciiTable
from torch.cuda import amp
from utils import threaded
from utils.confusion_matrix import ConfusionMatrix
from utils.plots import plot_images

from utils.torch_utils import ModelEMA
# Profilers
# from profilehooks import profile
# from line_profiler import profile
# from memory_profiler import profile

# Added on V0.3.0

from models import load_model
from utils.autobatcher import check_train_batch_size
from utils.logger import Logger
from utils.utils import (to_cpu, load_classes, print_environment_info, provide_determinism,
                         worker_seed_set, one_cycle, check_img_size,
                         labels_to_image_weights, labels_to_class_weights, tensor_to_np_array)
from utils.datasets import ListDataset
from utils.augmentations import AUGMENTATION_TRANSFORMS
from utils.parse_config import parse_data_config, parse_hyp_config
from utils.loss import compute_loss, fitness, training_fitness
from utils.writer import (csv_writer, img_writer_training, img_writer_evaluation,
                          log_file_writer, img_writer_eval_stats)
from utils.datasets_v2 import create_dataloader


def _create_data_loader(img_path, batch_size, img_size, n_cpu, multiscale_training=False):
    """Creates a DataLoader for training.

    :param img_path: Path to file containing all paths to training images.
    :type img_path: str
    :param batch_size: Size of each image batch
    :type batch_size: int
    :param img_size: Size of each image dimension for yolo
    :type img_size: int
    :param n_cpu: Number of cpu threads to use during batch generation
    :type n_cpu: int
    :param multiscale_training: Scale images to different sizes randomly
    :type multiscale_training: bool
    :return: Returns DataLoader
    :rtype: DataLoader
    """
    dataset = ListDataset(
        img_path,
        img_size=img_size,
        multiscale=multiscale_training,
        transform=AUGMENTATION_TRANSFORMS)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=n_cpu,
        pin_memory=True,
        collate_fn=dataset.collate_fn,
        worker_init_fn=worker_seed_set)
    return dataloader


# Python code to find and delete the oldest .pth file in the 'checkpoints/' directory
def find_and_del_last_ckpt():
    # Check if the file is a .pth file
    is_pth_file = False

    # Get the list of files in the 'checkpoints/' directory
    list_of_files = os.listdir('checkpoints/')

    # Get the full path of each file
    full_path = ["checkpoints/{0}".format(x) for x in list_of_files]

    # Find the oldest file based on creation time
    oldest_file = min(full_path, key=os.path.getctime)

    # Loop until a .pth file is found
    while not is_pth_file:
        if oldest_file.endswith('.pth'):
            is_pth_file = True
        else:
            # Remove the oldest file from the list
            full_path.remove(oldest_file)

            # Find the new oldest file
            oldest_file = min(full_path, key=os.path.getctime)

    # Delete the oldest .pth file
    os.remove(oldest_file)

# Python code to check and create necessary folders for logging, checkpoints, and output.

def check_folders():
    local_path = os.getcwd()
    # Check if logs folder exists
    logs_path_there = os.path.exists(local_path + "/logs/")
    if not logs_path_there:
        os.mkdir(local_path + "/logs/")
    # Check if logs/profiles folder exists
    logs_path_there = os.path.exists(local_path + "/logs/profiles/")
    if not logs_path_there:
        os.mkdir(local_path + "/logs/profiles/")
    # Check if checkpoints folder exists
    ckpt_path_there = os.path.exists(local_path + "/checkpoints/")
    if not ckpt_path_there:
        os.mkdir(local_path + "/checkpoints/")
    # Check if checkpoints/best folder exists
    ckpt_best_path_there = os.path.exists(local_path + "/checkpoints/best/")
    if not ckpt_best_path_there:
        os.mkdir(local_path + "/checkpoints/best/")
    # Check if output folder exists
    output_path_there = os.path.exists(local_path + "/output/")
    if not output_path_there:
        os.mkdir(local_path + "/output/")

@threaded()
def run_tensorboard(tracking_address):
    """
    This code defines a function to run Tensorboard for monitoring TensorFlow models.
    It uses the TensorBoard library to configure and launch Tensorboard with
    the specified tracking address. After launching, it prints the URL where Tensorboard is active.

    Example usage:
    run_tensorboard('/path/to/tracking/address')
    """
    tb = program.TensorBoard()
    tb.configure(argv=[None, '--logdir', tracking_address, '--bind_all'])
    url = tb.launch()
    print(f"-🎦- Tensorflow monitoring active on {url} ----")


def run(args, data_config, hyp_config, ver, clearml=None):
    '''
    211-339
    The code creates a model log folder and writes log files for a deep learning model.
    It also creates CSV files for training, evaluation, and validation. It sets up logging variables and
    loads configuration parameters for the model. The code also checks if certain folders exist and
    creates them if they do not. Finally, it initializes various variables and arrays for the model.
    '''
    date = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    if args.test_cycle is True:
        # Check folders
        check_folders()
    try:

        if args.optimizer != None:
            hyp_config['optimizer'] = args.optimizer
        if args.scheduler != None:
            hyp_config['scheduler'] = args.scheduler
        if args.seed != -1:
            provide_determinism(args.seed)

        train_path = data_config["train"]
        valid_path = data_config["valid"]
        class_names = load_classes(data_config["names"])
        num_classes = len(class_names)

        if args.name != None:
            model_name = args.name
        else:
            model_name = data_config["model_name"]
            if model_name == '':
                model_name = str(date)
            else:
                model_name = model_name + '_' + str(date)
        local_path = os.getcwd()
        model_logs_path = os.path.join(local_path, args.logdir, model_name)
        # Create model named log folder in logs folder

        # Check if logs folder exists
        logs_path_there = os.path.exists(model_logs_path)
        if not logs_path_there:
            os.mkdir(model_logs_path + '/')
        model_imgs_logs_path = '/images/'
        logs_img_path_there = os.path.exists(model_logs_path + model_imgs_logs_path)
        if not logs_img_path_there:
            os.mkdir(model_logs_path + model_imgs_logs_path)
            model_imgs_logs_path = model_logs_path + model_imgs_logs_path
        model_ckpt_path = '/checkpoints/'
        model_ckpt_path_there = os.path.exists(model_logs_path + model_ckpt_path)
        if not model_ckpt_path_there:
            os.mkdir(model_logs_path + model_ckpt_path)
            model_ckpt_logs_path = model_logs_path + model_ckpt_path
        # Create and write model log files
        model_logfile_path = model_logs_path + '/' + model_name + "_logfile" + ".txt"
        # Create new log file
        f = open(model_logfile_path, "w")
        f.close()
        log_file_writer("Software version: " + ver, model_logfile_path)
        log_file_writer(f"Command line arguments: {args}", model_logfile_path)
        if args.verbose:
            print(f'Class names: {class_names}')
            log_file_writer(f"Class names: {class_names}", model_logfile_path)
        print_environment_info(ver, model_logfile_path)
        logger = Logger(model_logs_path)  # Tensorboard logger
        run_tensorboard(model_logs_path)
        gpu = args.gpu
        auto_eval = True
        best_training_fitness = 999999
        best_fitness = 0.0
        checkpoints_saved = 0
        device = torch.device("cpu")
        cuda_available = False
        mem = 0.0
        exec_time = 0
        do_auto_eval = False
        if args.warmup:
            warmup_run = True
        else:
            warmup_run = False
        start_epoch, train_fitness = 0, 0
        clearml_run = False
        # Get model weight eval parameters
        # Access the parameters from the config file
        w_train = []
        w = []
        w_train_str = hyp_config['w_train'].strip('][').split(', ')
        w_str = hyp_config['w'].strip('][').split(', ')
        lr_restart = False

        for i in w_train_str:
            w_train.append(float(i))
        for i in w_str:
            w.append(float(i))
        # #################
        # Create Logging variables
        # #################

        # Matplotlib arrays
        iou_loss_array = np.array([])
        obj_loss_array = np.array([])
        cls_loss_array = np.array([])
        batch_loss_array = np.array([])
        batches_array = np.array([])
        train_loss_array = np.array([])
        lr_array = np.array([])
        eval_epoch_array = np.array([])
        precision_array = np.array([])
        recall_array = np.array([])
        m_ap_array = np.array([])
        f1_array = np.array([])

        # ap_cls_array = np.array([])
        curr_fitness_array = np.array([])
        train_fitness_array = np.array([])

        # last_opt_step = -1
        # Define the maximum gradient norm for clipping
        max_grad_norm = 1.0

        ################
        # Create CSV files - version 0.3.8
        ################

        # Create training csv file
        # header = ['Iterations', 'Iou Loss', 'Object Loss', 'Class Loss', 'Loss', 'Learning Rate']
        header = ['Iterations', 'Iou Loss', 'Object Loss', 'Class Loss', 'Loss', 'Mean loss',
                  'Learning Rate']
        csv_writer(header, model_logs_path + "/" + model_name + "_training_plots.csv", 'a')

        # Create evaluation csv file
        header = ['Epoch', 'Epochs', 'Precision', 'Recall', 'mAP', 'F1', 'Fitness']
        csv_writer(header, model_logs_path + "/" + model_name + "_evaluation_plots.csv", 'a')

        # Create validation csv file
        header = ['Index', 'Class', 'AP']
        csv_writer(header, f"checkpoints/best/{model_name}_eval_stats.csv", 'a')
        '''
        346-430
        This code is used for training a model. It includes the following steps:
        1. Create a ClearML task.
        2. Check if GPU is available and set the device accordingly.
        3. Load the model with specified parameters.
        4. Log the hyperparameters to ClearML.
        5. Print the model summary if verbose mode is enabled.
        6. Calculate the batch size based on the model's hyperparameters.
        7. Perform batch size calculations and set necessary variables.
        8. Scale the weight decay based on the batch size.
        '''
        ################
        # Create ClearML task - version 0.3.0
        ################

        if clearml is not None:
            # Access the parameters from the config file
            proj_name = clearml.get('clearml', 'proj_name')
            # task_name = config.get('clearml', 'task_name')
            offline = clearml.get('clearml', 'offline')
            if clearml.get('clearml', 'clearml_save_last') == "True":
                clearml_save_last = True
            else:
                clearml_save_last = False
            if clearml.get('clearml', 'clearml_run') == "True":
                clearml_run = True
            else:
                clearml_run = False

            if clearml_run:
                import clearml
                task_name = model_name
                if offline == "True":
                    # Use the set_offline class method before initializing a Task
                    clearml.Task.set_offline(offline_mode=True)
                # Create a new task
                task = clearml.Task.init(project_name=proj_name, task_name=task_name, auto_connect_frameworks={
                    'matplotlib': False, 'tensorflow': False, 'tensorboard': False, 'pytorch': True,
                    'xgboost': False, 'scikit': True, 'fastai': False, 'lightgbm': False,
                    'hydra': False, 'detect_repository': True, 'tfdefines': False, 'joblib': False,
                    'megengine': False, 'jsonargparse': True, 'catboost': False})
                # Log model configurations
                task.connect(args)
                # Instantiate an OutputModel with a task object argument
                clearml.OutputModel(task=task, framework="PyTorch")

        if gpu != -1:
            if torch.cuda.is_available() is True:
                device = torch.device("cuda")
                cuda_available = True
            else:
                device = torch.device("cpu")
                cuda_available = False
        print(f'---- Using cuda device - {device} ----')
        log_file_writer(f'Using cuda device - {device}', model_logfile_path)

        # ############
        # Create model - Updated on V0.4.0
        # ############
        model = load_model(args.model, hyp_config, gpu, args.pretrained_weights)

        # ############
        # Freeze model layers
        # ############
        # -- Not implemented --

        # ############
        # Log hyperparameters to clearml
        # ############
        if clearml_run:
            task.connect_configuration(model.hyperparams)
        log_file_writer(f"Model hyperparameters: {model.hyperparams}", model_logfile_path)

        # Print model
        if args.verbose:
            summary(model, input_size=(3, model.hyperparams['height'], model.hyperparams['height']))

        # ############
        # Batch size calculation - V0.3.1
        # ############

        batch_size = model.hyperparams['batch']
        try:
            batch_size = check_train_batch_size(model, model.hyperparams['height'], cuda_available)
            sub_div = 1
        except:
            batch_size = model.hyperparams['batch']
            sub_div = model.hyperparams['subdivisions']

        mini_batch_size = batch_size // sub_div
        nbs = 64  # nominal batch size
        accumulate = max(round(nbs / batch_size), 1)  # accumulate loss before optimizing
        hyp_config['weight_decay'] = float(
            hyp_config['weight_decay']) * batch_size * accumulate / nbs  # scale weight_decay

        # 441 - 538
        # This code defines an optimizer and handles smart resume functionality for training a model.
        # It first categorizes the model's parameters into three groups: biases, weights with weight decay, and other parameters.
        # It then selects the optimizer based on the specified or default optimizer and initializes it with the appropriate parameters.
        # If the specified optimizer is not implemented, it reverts to using the SGD optimizer.
        # Additionally, it adds the weight decay to the weight parameters and includes the biases in the optimizer.
        # The code also handles smart resume by loading a pretrained model if specified and setting the learning rate accordingly.

        # ################
        # Create optimizer - V0.4
        # ################

        pg0, pg1, pg2 = [], [], []  # optimizer parameter groups
        for k, v in dict(model.named_parameters()).items():
            if '.bias' in k:
                pg2.append(v)  # biases
            elif 'Conv2d.weight' in k:
                pg1.append(v)  # apply weight_decay
            elif 'm.weight' in k:
                pg1.append(v)  # apply weight_decay
            elif 'w.weight' in k:
                pg1.append(v)  # apply weight_decay
            else:
                pg0.append(v)  # all else

        if args.optimizer != None:
            req_optimizer = args.optimizer
        else:
            req_optimizer = hyp_config['optimizer']
        params = [p for p in model.parameters() if p.requires_grad]
        implemented_optimizers = ["adamw", 'sgd', "rmsprop", "adadelta", "adamax", "adam"]
        if req_optimizer in implemented_optimizers:
            if req_optimizer == "adamw":
                optimizer = optim.AdamW(
                    params,
                    lr=float(hyp_config['lr0']),
                    betas=(float(hyp_config['momentum']), 0.999),
                    amsgrad=True
                )
            elif req_optimizer == "sgd":
                optimizer = optim.SGD(
                    pg0,
                    lr=float(hyp_config['lr0']),
                    momentum=float(hyp_config['momentum']),
                    nesterov=True,
                )
            elif req_optimizer == "rmsprop":
                optimizer = optim.RMSprop(
                    pg0,
                    lr=float(hyp_config['lr0']),
                    momentum=float(hyp_config['momentum'])
                )

            elif req_optimizer == "adam":
                optimizer = optim.Adam(
                    pg0,
                    lr=float(hyp_config['lr0']),
                    betas=(float(hyp_config['momentum']), 0.999),
                )
            elif req_optimizer == "adadelta":
                optimizer = optim.Adadelta(
                    pg0,
                    lr=float(hyp_config['lr0']),
                )
            elif req_optimizer == "adamax":
                optimizer = optim.Adamax(
                    pg0,
                    lr=float(hyp_config['lr0']),
                    betas=(float(hyp_config['momentum']), 0.999),
                )

        else:
            print("- ⚠ - Unknown optimizer. Reverting into SGD optimizer.")
            log_file_writer(f"- ⚠ - Unknown optimizer. Reverting into SGD optimizer.", model_logfile_path)
            optimizer = optim.SGD(
                pg0,
                lr=float(hyp_config['lr0']),
                momentum=float(hyp_config['momentum']),
                nesterov=True,
            )
            model.hyperparams['optimizer'] = 'sgd'
        if req_optimizer == 'sgd':
            optimizer.add_param_group(
                {'params': pg1, 'weight_decay': hyp_config['weight_decay']})  # add pg1 with weight_decay
            optimizer.add_param_group({'params': pg2})  # add pg2 (biases)
            print(f'---- Optimizer groups: {len(pg2)} .bias, {len(pg1)} conv.weight, {len(pg0)} other ----')
        del pg0, pg1, pg2

        # #################
        # Smart resume - V0.4
        # #################
        pretrained = args.pretrained_weights.endswith('.pth')
        if pretrained:
            #resume = True
            warmup_run = False
            lr_restart = True
            lr = float(hyp_config['lr0'])
            # Set learning rate
            for g in optimizer.param_groups:
                g['lr'] = lr
            ckpt = torch.load(args.pretrained_weights, map_location=device)  # load checkpoint
            # state_dict = {k: v for k, v in ckpt.items() if model.state_dict()[k].numel() == v.numel()}
            # Optimizer

            del ckpt  # , state_dict

        '''
        547-699
        The code is a part of a training script for a machine learning model. 
        It creates data loaders for training and validation datasets. 
        It also sets up the scheduler for adjusting the learning rate during training based on the chosen optimizer. 
        The code supports various scheduler options such as CosineAnnealingLR, ExponentialLR, ReduceLROnPlateau, etc. 
        The maximum number of iterations and warm-up iterations are calculated based on the provided hyperparameters. 
        The code also includes some commented out sections related to autoanchor and pre-training routines.
        '''

        # #################
        # Create Dataloader - V0.4
        # #################
        # Image sizes
        image_dims = [int(model.hyperparams['width']), int(model.hyperparams['height'])]
        gs = 64  # int(max(model.stride))  # grid size (max stride)
        imgsize, test_images = [check_img_size(x, gs) for x in image_dims]  # verify imgsz are gs-multiples
        # Trainloader
        dataloader, dataset = create_dataloader(train_path, imgsize, batch_size, gs, args, class_names,
                                                model_imgs_logs_path,
                                                hyp=hyp_config, augment=True, cache=False, rect=False,
                                                rank=-1, world_size=1, workers=int(args.n_cpu))

        '''
        validation_dataloader = create_dataloader(valid_path, imgsize, batch_size, gs, args, 
                                        class_names, model_imgs_logs_path,
                                       hyp=hyp_config, augment=True,cache=False, rect=True,
                                       rank=-1, world_size=1, workers=int(args.n_cpu))  # testloader
        '''
        validation_dataloader = _create_validation_data_loader(
            valid_path,
            mini_batch_size,
            model.hyperparams['height'],
            args.n_cpu)

        warmup_epochs = float(hyp_config['warmup_epochs'])
        if warmup_epochs > 5.0:
            warmup_epochs = 5.0
        num_batches = len(dataloader)  # number of batches
        warmup_num = max(
            round(warmup_epochs * num_batches), 100)  # number of warmup iterations, max(5 epochs, 100 iterations)
        if warmup_num > num_batches * args.epochs:
            warmup_num = float(num_batches * args.epochs / 2)  # limit warmup to < 1/2 of training
        print(f'- 🔥 - Number of calculated warmup iterations: {warmup_num} ----')
        max_batches = len(class_names) * int(model.hyperparams['max_batches_factor'])
        num_steps = num_batches * args.epochs

        print(f"- ⚠ - Maximum number of iterations - {max_batches}")
        log_file_writer(f"Maximum batch size: {max_batches}", model_logfile_path)
        # #################
        # Use autoanchor -> Not implemented yet
        # #################

        '''
        if not resume:
                if not args.noautoanchor:
                    check_anchors(dataset, model=model, thr=hyp['anchor_t'], imgsz=imgsz)  # run AutoAnchor
                model.half().float()  # pre-reduce anchor precision

            callbacks.run('on_pretrain_routine_end', labels, names)
        '''

        # #################
        # Scheduler selector - V0.3.18
        # #################

        if args.scheduler != None:
            req_scheduler = args.scheduler
        else:
            req_scheduler = hyp_config['lr_scheduler']
        implemented_schedulers = ['CosineAnnealingLR', 'ChainedScheduler',
                                  'ExponentialLR', 'ReduceLROnPlateau', 'ConstantLR',
                                  'CyclicLR', 'OneCycleLR', 'LambdaLR', 'MultiplicativeLR',
                                  'StepLR', 'MultiStepLR', 'LinearLR', 'PolynomialLR', 'CosineAnnealingWarmRestarts']
        if req_scheduler in implemented_schedulers:
            lf = lambda x: ((1 + math.cos(x * math.pi / args.epochs)) / 2) * (1 - float(hyp_config['lrf'])) + float(
                hyp_config['lrf'])  # cosine

            # CosineAnnealingLR
            if req_scheduler == 'CosineAnnealingLR':
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer,
                    T_max=int(num_steps / 10),
                    eta_min=float(hyp_config['lr0']) / 10000,
                    verbose=False)
            # ChainedScheduler
            elif req_scheduler == 'ChainedScheduler':
                scheduler1 = ConstantLR(optimizer, factor=0.5, total_iters=5,
                                        verbose=False)
                scheduler2 = ExponentialLR(optimizer, gamma=float(hyp_config['dec_gamma']), verbose=False)
                scheduler = torch.optim.lr_scheduler.ChainedScheduler([scheduler1, scheduler2])
            elif req_scheduler == 'ExponentialLR':
                scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=float(hyp_config['dec_gamma']),
                                                                   verbose=False)
            elif req_scheduler == 'ReduceLROnPlateau':
                minimum_lr = float(hyp_config['lr0']) / float(hyp_config['lrf'])
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer,
                    'min',
                    patience=int(args.epochs / 10),
                    min_lr=minimum_lr,
                    verbose=False)
            elif req_scheduler == 'ConstantLR':
                scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer, factor=0.5,
                                                                total_iters=int(args.evaluation_interval),
                                                                verbose=False)
            elif req_scheduler == 'CyclicLR':
                scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer,
                                                              base_lr=float(hyp_config['lr0']) / float(
                                                                  hyp_config['lrf']),
                                                              max_lr=float(hyp_config['lr0']), cycle_momentum=True,
                                                              verbose=False,
                                                              mode='exp_range')  # mode (str): One of {triangular, triangular2, exp_range}.
            elif req_scheduler == 'OneCycleLR':
                scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=float(hyp_config['lrf']),
                                                                steps_per_epoch=len(dataloader),
                                                                epochs=int(args.epochs))
            elif req_scheduler == 'LambdaLR':
                scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,
                                                              lr_lambda=lf,
                                                              verbose=False)  # plot_lr_scheduler(optimizer, scheduler, epochs)
            elif req_scheduler == 'MultiplicativeLR':
                lf = one_cycle(1, float(hyp_config['lrf']), args.epochs)  # cosine 1->hyp['lrf']
                scheduler = torch.optim.lr_scheduler.MultiplicativeLR(optimizer, lr_lambda=lf)
            elif req_scheduler == 'StepLR':
                scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=int(args.epochs) / 10,
                                                            gamma=0.1)  # Step size -> epochs
            elif req_scheduler == 'MultiStepLR':
                scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30, 80],
                                                                 gamma=0.1)  # milestones size -> epochs
            elif req_scheduler == 'LinearLR':
                scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.5, total_iters=int(
                    args.epochs) / 2)  # total_iters size -> epochs
            elif req_scheduler == 'PolynomialLR':
                scheduler = torch.optim.lr_scheduler.PolynomialLR(optimizer, total_iters=int(args.epochs),
                                                                  power=1.0)  # total_iters size -> epochs
            elif req_scheduler == 'CosineAnnealingWarmRestarts':
                scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=int(num_steps / 10),
                                                                                 eta_min=float(
                                                                                     hyp_config['lr0']) / float(
                                                                                     hyp_config[
                                                                                         'lrf']))  # total_iters size -> epochs
        else:
            print("- ⚠ - Unknown scheduler! Reverting to LambdaLR")
            req_scheduler = 'LambdaLR'
            lf = lambda x: (1 - x / args.epochs) * (1.0 - float(hyp_config['lrf'])) + float(hyp_config['lrf'])
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,
                                                          lr_lambda=lf, verbose=False)
        print(
            f"- ⚠ - Using {req_optimizer} - optimizer with {req_scheduler} - LR scheduler")
        log_file_writer(f"Using {req_optimizer} - optimizer with {req_scheduler} - LR scheduler", model_logfile_path)

        scheduler.last_epoch = start_epoch - 1  # do not move

        decay_schedulers = ['ConstantLR', 'ExponentialLR', 'MultiplicativeLR', 'StepLR', 'MultiStepLR', 'LinearLR',
                            'PolynomialLR', 'ReduceLROnPlateau']
        if req_scheduler in decay_schedulers:
            # Set learning rate for decaying schedulers
            lr = float(hyp_config['lr0'])
            for g in optimizer.param_groups:
                g['lr'] = lr

        # #################
        # Use ModelEMA - V0.x.xx -> Not implemented correctly
        # #################
        ema = ModelEMA(model) if args.ema != -1 else None

        # #################
        # Create GradScaler - V 0.3.14
        # #################
        # Creates a GradScaler once at the beginning of training.
        scaler = torch.cuda.amp.GradScaler(enabled=cuda_available)

        # #################
        # SyncBatchNorm - V 0.3.14 -> not needed in current state, but is basis if multi-gpu support is created
        # #################
        if args.sync_bn != -1 and torch.cuda.is_available() is True:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(device)
            log_file_writer(f'Using SyncBatchNorm()', model_logfile_path)

        # Model parameters
        hyp_config['cls'] = float(
            hyp_config['cls']) * num_classes / 80.  # scale coco-tuned hyp['cls'] to current dataset
        model.nc = num_classes  # attach number of classes to model
        model.hyp = hyp_config  # attach hyperparameters to model
        model.gr = 1.0  # iou loss ratio (obj_loss = 1.0 or iou)
        # model.gr = args.iou_thres
        model.class_weights = labels_to_class_weights(dataset.labels, num_classes).to(device)  # attach class weights
        model.names = class_names

        # #################
        # ConfusionMatrix - V 0.4.5F
        # #################
        confusion_matrix = ConfusionMatrix(nc=len(class_names))

        # skip epoch zero, because then the calculations for when to evaluate/checkpoint makes more intuitive sense
        # e.g. when you stop after 30 epochs and evaluate every 10 epochs then the evaluations happen after: 10,20,30
        # instead of: 0, 10, 20

        print(f"\n- 🔛 - Starting Model {model_name} training... ----")

        torch.save(model, f'{model_ckpt_logs_path}/{model_name}_init.pt')
        # Modded on V0.4

        for epoch in range(1, args.epochs + 1):
            epoch_start = time.time()
            if epoch > 1:
                print(
                    f'- ⏳ - Estimate when all {args.epochs} epochs are done: {round((exec_time * (args.epochs - epoch)) / 3600, 2)} hours ----')
            if warmup_run:
                print(f'- 🔥 - Running warmup cycle ----')
            if torch.cuda.is_available():
                mem = '%.3gG' % (torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0)  # (GB)
                print(f'---- GPU Memory usage: {mem} ----')
            model.train()  # Set model to training mode
            mloss = torch.zeros(4, device=device)  # mean losses
            # https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html
            # for param in model.parameters():
            #    param.grad = None

            for batch_i, (imgs, targets, paths, _) in enumerate(tqdm.tqdm(dataloader, desc=f"Training Epoch {epoch}")):
                optimizer.zero_grad()
                optimizer.step()
                batches_done = len(dataloader) * epoch + batch_i
                integ_batch_num = batch_i + num_batches * epoch  # number integrated batches (since train start)

                imgs = imgs.to(device,
                               non_blocking=True).float() / 255  # -> causes overflow sometimes # uint8 to float32, 0-255 to 0.0-1.0
                # imgs = imgs.to(device, non_blocking=True)

                ###########
                # Warmup - 0.4
                ###########
                if warmup_run:
                    if integ_batch_num <= warmup_num:
                        # scaler.step(optimizer)ät
                        x_interp = [0, warmup_num]
                        # accumulate = max(1, np.interp(integ_batch_num, x_interp, [1, num_batches / batch_size]).round())
                        # Simplified version
                        accumulate = max(1, min(integ_batch_num, num_batches / batch_size))
                        for j, x in enumerate(optimizer.param_groups):
                            # bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
                            if req_scheduler == "ReduceLROnPlateau":
                                # Burn in
                                x['lr'] *= (batches_done / warmup_num)
                            else:
                                # x['lr'] = np.interp(integ_batch_num, x_interp,[hyp_config['warmup_bias_lr'] if j == 2 else 0.0, x['initial_lr'] * lf(epoch)])
                                conditions = [integ_batch_num < warmup_num, integ_batch_num >= warmup_num]
                                choices = [0.0, x['initial_lr'] * epoch]  # ReduceLROnPlateau -> KeyError: 'initial_lr'
                                x['lr'] = np.select(conditions, choices, default=float(hyp_config['warmup_bias_lr']))

                            if 'momentum' in x:
                                x['momentum'] = np.interp(integ_batch_num, x_interp,
                                                          [float(hyp_config['warmup_momentum']),
                                                           float(hyp_config['momentum'])])

                else:
                    warmup_run = False
                    if not lr_restart:
                        lr_restart = True
                        # Get learning rate
                        lr = float(hyp_config['lr0'])
                        # Set learning rate
                        for g in optimizer.param_groups:
                            g['lr'] = lr
                # Multi-scale not implemented -> snippet below is just an test example
                '''
                # Multi-scale
                if opt.multi_scale:
                    sz = random.randrange(imgsz * 0.5, imgsz * 1.5 + gs) // gs * gs  # size
                    sf = sz / max(imgs.shape[2:])  # scale factor
                    if sf != 1:
                        ns = [math.ceil(x * sf / gs) * gs for x in
                              imgs.shape[2:]]  # new shape (stretched to gs-multiple)
                        imgs = F.interpolate(imgs, size=ns, mode='bilinear', align_corners=False)
                '''
                ###############
                # Forward
                ###############
                with amp.autocast(enabled=cuda_available):
                    pred = model(imgs)  # forward
                    loss, loss_items = compute_loss(pred, targets.to(device), model)  # loss scaled by batch_size
                mloss = (mloss * batch_i + loss_items) / (batch_i + 1)  # update mean losses
                mloss_mean = np.mean(tensor_to_np_array(mloss))
                # Apply gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

                ###############
                # Backward
                ###############
                scaler.scale(loss).backward()

                ###############
                # Run optimizer
                ###############
                # Optimize
                if integ_batch_num % accumulate == 0:
                    scaler.step(optimizer)  # optimizer.step
                    scaler.update()
                    optimizer.zero_grad()
                    model.zero_grad()  # Test
                    if ema:
                        ema.update(model)



                # Scheduler
                # scheduler.get_last_lr()
                if req_scheduler != 'ReduceLROnPlateau':
                    scheduler.step()
                else:
                    scheduler.step(loss)
                    if not warmup_run:
                        # Get learning rate
                        lr = float(optimizer.param_groups[0]['lr'])
                        # Set learning rate
                        for g in optimizer.param_groups:
                            g['lr'] = lr
                lr = [x['lr'] for x in optimizer.param_groups]  # for tensorboard
                # print(f'DEBUG - Scheduler last lr: {scheduler.get_last_lr()}  <-> Optimizer lr: {lr}')
                # mAP
                if loss_items.dim() != 0:
                    # ############
                    # Log progress
                    # ############
                    if args.verbose:
                        print(AsciiTable(
                            [
                                ["Type", "Value"],
                                ["IoU loss", float(loss_items[0])],
                                ["Object loss", float(loss_items[1])],
                                ["Class loss", float(loss_items[2])],
                                ["Loss", float(loss_items[3])],
                                ["Batch Loss", float(loss.item())],
                                # ["Batch loss", to_cpu(mloss).item()],
                            ]).table)

                    # Tensorboard logging
                    tensorboard_log = [
                        ("train/iou_loss", float(loss_items[0])),
                        ("train/obj_loss", float(loss_items[1])),
                        ("train/class_loss", float(loss_items[2])),
                        ("train/loss", float(loss_items[3])),
                        ("train/batch loss", float(loss.item())),

                    ]
                    logger.list_of_scalars_summary(tensorboard_log, batches_done)
                    # Tensorflow logger - learning rate V0.3.4I
                    logger.scalar_summary("train/learning rate", np.mean(lr), batches_done)

                    model.seen += imgs.size(0)

                    # ############
                    # ClearML progress logger - V0.3.3
                    # ############
                    if clearml_run:
                        task.logger.report_scalar(title="Train/Losses", series="IoU loss", iteration=batches_done,
                                                  value=float(loss_items[0]))
                        task.logger.report_scalar(title="Train/Losses", series="Object loss", iteration=batches_done,
                                                  value=float(loss_items[1]))
                        task.logger.report_scalar(title="Train/Losses", series="Class loss", iteration=batches_done,
                                                  value=float(loss_items[2]))
                        task.logger.report_scalar(title="Train/Losses", series="Loss", iteration=batches_done,
                                                  value=float(loss_items[3]))
                        task.logger.report_scalar(title="Train/Losses", series="Batch loss", iteration=batches_done,
                                                  value=loss.item())
                        task.logger.report_scalar(title="Train/Lr", series="Learning rate", iteration=batches_done,
                                                  value=np.mean(lr))

                # ############
                # Log training progress writers
                # ############
                #
                # training csv writer
                if loss_items.dim() > 0:
                    lr_float = format(float(np.mean(lr)), '.7f')
                    data = [batches_done,
                            float(loss_items[0]),  # Iou Loss
                            float(loss_items[1]),  # Object Loss
                            float(loss_items[2]),  # Class Loss
                            float(loss_items[3]),  # Loss
                            float(loss.item()),  # Batch loss
                            float(mloss_mean),  # Mean Loss
                            (lr_float)  # Learning rate
                            ]
                    # header = ['Iterations', 'Iou Loss', 'Object Loss', 'Class Loss', 'Loss', 'Batch loss','Mean loss','Learning Rate']
                    csv_writer(data, model_logs_path + "/" + model_name + "_training_plots.csv", 'a')

                    # ############
                    # ClearML csv reporter logger - V0.3.6
                    # ############
                    if clearml_run:
                        # Report table - CSV from path
                        csv_url = model_logs_path + "/" + model_name + "_training_plots.csv"
                        task.logger.report_table(
                            "Training plots",
                            "training_plots.csv",
                            iteration=batches_done,
                            url=csv_url
                        )

                    # img writer
                    batches_array = np.concatenate((batches_array, np.array([batches_done])))
                    iou_loss_array = np.concatenate((iou_loss_array, np.array([float(loss_items[0])])))
                    obj_loss_array = np.concatenate((obj_loss_array, np.array([float(loss_items[1])])))
                    cls_loss_array = np.concatenate((cls_loss_array, np.array([float(loss_items[2])])))
                    train_loss_array = np.concatenate((train_loss_array, np.array([float(loss_items[3].item())])))
                    batch_loss_array = np.concatenate((batch_loss_array, np.array([float(loss.item())])))
                    lr_array = np.concatenate((lr_array, np.array([lr_float])))
                    img_writer_training(iou_loss_array, obj_loss_array, cls_loss_array, train_loss_array, lr_array,
                                        batch_loss_array,
                                        batches_array,
                                        model_logs_path + '/' + model_name)
            # #############
            # Save progress -> changed on version 0.3.11F to save every eval epoch
            # #############
            # Reason to eval epoch change: uploads get stucked when using clearml and larger models
            #
            if epoch % args.evaluation_interval == 0:


                # Save last model to checkpoint file
                # Updated on version 0.3.0 to save only last
                checkpoint_path = f"{model_ckpt_logs_path}/{model_name}_ckpt_last.pth"
                print(f"- ⏺ - Saving last checkpoint to: '{checkpoint_path}' ----")
                torch.save(model.state_dict(), checkpoint_path)
                checkpoints_saved += 1

                ############################
                # ClearML last model update - V 0.3.7 -> changed on version 0.3.11F to save every eval epoch
                ############################
                if clearml_run and clearml_save_last:
                    task.update_output_model(model_path=f"{model_ckpt_logs_path}/{model_name}_ckpt_last.pth")

            if auto_eval is True and loss_items.dim() > 0:
                # #############
                # Training fitness evaluation
                # Calculate weighted loss -> smaller losses = better training fitness
                # #############
                print("\n- 🔄 - Auto evaluating model on training metrics ----")
                training_evaluation_metrics = [
                    float(loss_items[0]),  # Iou Loss
                    float(loss_items[1]),  # Object Loss
                    float(loss_items[2]),  # Class Loss
                    float(loss_items[3]),  # Loss
                ]
                # Updated on version 0.3.12
                fi_train = training_fitness(np.array(training_evaluation_metrics).reshape(1, -1), w_train)
                train_fitness = float(fi_train[0])
                logger.scalar_summary("fitness/training", train_fitness, epoch)
                if fi_train < best_training_fitness:
                    print(
                        f"- ✅ - Auto evaluation result: New best training fitness {fi_train}, old best {best_training_fitness} ----")
                    best_training_fitness = fi_train
                    do_auto_eval = True
                else:
                    print(
                        f"- ❎ - Auto evaluation result: Training fitness {fi_train}, best {best_training_fitness} ----")

                # ############
                # ClearML training fitness logger - V0.3.4
                # ############
                if clearml_run:
                    task.logger.report_scalar(title="Training", series="Fitness", iteration=epoch,
                                              value=float(fi_train[0]))

            # ########
            # Evaluate
            # ########
            if epoch % int(args.evaluation_interval) == 0 or do_auto_eval:
                do_auto_eval = False
                # Do evaluation on every epoch for better logging
                print("\n- 🔄 - Evaluating Model ----")
                # Evaluate the model on the validation set
                metrics_output, eval_outputs, eval_targets = _evaluate(
                    model,
                    validation_dataloader,
                    class_names,
                    model_imgs_logs_path,
                    epoch,
                    args.draw,
                    img_size=model.hyperparams['height'],
                    iou_thres=args.iou_thres,
                    conf_thres=args.conf_thres,
                    nms_thres=args.nms_thres,
                    verbose=args.verbose,
                    device=device,

                )


                # Plot
                # if args.evaluation_interval % epoch == 0 and args.verbose:
                #if args.draw:
                #    f = f'{model_logs_path}/images/train_batch{integ_batch_num}.jpg'  # filename
                #    plot_images(images=imgs, targets=eval_targets, paths=model_logs_path, fname=f)
                #    # if tb_writer:
                #    #     tb_writer.add_image(f, result, dataformats='HWC', global_step=epoch)
                #    #     tb_writer.add_graph(model, imgs)  # add model to tensorboard

                # Create confusion matrix -> changed in version 0.4.7
                #confusion_matrix.generate_batch_data(eval_outputs, eval_targets)
                #confusion_matrix.plot(True, model_imgs_logs_path, class_names)

                if metrics_output is not None:
                    precision, recall, AP, f1, ap_class = metrics_output
                    evaluation_metrics = [
                        precision.mean(),
                        recall.mean(),
                        AP.mean(),
                        f1.mean(),
                        ap_class.mean()
                    ]

                    # Log the evaluation metrics
                    logger.scalar_summary("validation/precision", float(precision.mean()), epoch)
                    logger.scalar_summary("validation/recall", float(recall.mean()), epoch)
                    logger.scalar_summary("validation/mAP", float(AP.mean()), epoch)
                    logger.scalar_summary("validation/f1", float(f1.mean()), epoch)

                    # ############
                    # ClearML validation logger - V0.3.3
                    # ############
                    if clearml_run:
                        task.logger.report_scalar(title="Validation", series="Precision", iteration=epoch,
                                                  value=float(precision.mean()))
                        task.logger.report_scalar(title="Validation", series="Recall", iteration=epoch,
                                                  value=float(recall.mean()))
                        task.logger.report_scalar(title="Validation", series="mAP", iteration=epoch,
                                                  value=float(AP.mean()))
                        task.logger.report_scalar(title="Validation", series="F1", iteration=epoch,
                                                  value=float(f1.mean()))
                    # ############
                    # Current fitness calculation - V0.3.6B
                    # ############
                    # Updated on version 0.3.12
                    fi = fitness(np.array(evaluation_metrics).reshape(1, -1),
                                 w)  # weighted combination of [P, R, mAP@0.5, f1]
                    curr_fitness = float(fi[0])
                    curr_fitness_array = np.concatenate((curr_fitness_array, np.array([curr_fitness])))
                    logger.scalar_summary("fitness/model", curr_fitness, epoch)
                    train_fitness_array = np.concatenate((train_fitness_array, np.array([train_fitness])))
                    # logger.scalar_summary("fitness/training", float(fi_train), epoch)
                    print(
                        f"- ➡ - Checkpoint fitness: '{round(curr_fitness, 6)}' (Current best fitness: {round(best_fitness, 6)}) ----")

                    if clearml_run:
                        # ############
                        # ClearML fitness logger - V0.3.3
                        # ############
                        task.logger.report_scalar(title="Checkpoint", series="Fitness", iteration=epoch,
                                                  value=curr_fitness)
                    # img writer - evaluation
                    eval_epoch_array = np.concatenate((eval_epoch_array, np.array([epoch])))
                    precision_array = np.concatenate((precision_array, np.array([precision.mean()])))
                    recall_array = np.concatenate((recall_array, np.array([recall.mean()])))
                    m_ap_array = np.concatenate((m_ap_array, np.array([AP.mean()])))
                    f1_array = np.concatenate((f1_array, np.array([f1.mean()])))
                    img_writer_evaluation(precision_array, recall_array, m_ap_array, f1_array,
                                          curr_fitness_array, train_fitness_array, eval_epoch_array,
                                          model_logs_path + '/' + model_name)

                    if curr_fitness > best_fitness and epoch:
                        best_fitness = curr_fitness
                        checkpoint_path = f"{model_ckpt_logs_path}/{model_name}_ckpt_best.pth"
                        print(f"- ⭐ - Saving best checkpoint to: '{checkpoint_path}'  ----")
                        torch.save(model.state_dict(), checkpoint_path)

                        #Make a copy of best checkpoint confusion matrix
                        shutil.copyfile(f'{model_imgs_logs_path}/confusion_matrix_last.png',
                                        f'{model_imgs_logs_path}/confusion_matrix_best.png')
                        ############################
                        # ClearML model update - V 3.0.0
                        ############################
                        if clearml_run:
                            task.update_output_model(model_path=f"{model_ckpt_logs_path}/{model_name}_ckpt_best.pth")

                        ############################
                        # Save best checkpoint evaluation stats into csv - V0.3.8
                        #############################

                        precision, recall, AP, f1, ap_class = metrics_output
                        # Gets class AP and mean AP
                        # print('ap cls',ap_class)
                        # print('AP',AP)
                        # print(class_names)
                        csv_writer("", f"{model_logs_path}/{model_name}_eval_stats.csv", 'w')
                        eval_stats_class_array = np.array([])
                        eval_stats_ap_array = np.array([])

                        for i, c in enumerate(ap_class):
                            data = [c,  # Class index
                                    class_names[i],  # Class name
                                    "%.5f" % AP[i],  # Class AP
                                    ]
                            eval_stats_class_array = np.concatenate(
                                (eval_stats_class_array, np.array([class_names[i]])))
                            eval_stats_ap_array = np.concatenate((eval_stats_ap_array, np.array([AP[i]])))

                            logger.scalar_summary(f"validation/class/{class_names[i]}", round(float(AP[i]), 5), epoch)
                            csv_writer(data, f"{model_logs_path}/{model_name}_eval_stats.csv", 'a')

                        # Write mAP value as last line
                        data = ["--",  #
                                'mAP',  #
                                str(round(AP.mean(), 5)),
                                ]
                        csv_writer(data, f"{model_logs_path}/{model_name}_eval_stats.csv", 'a')
                        img_writer_eval_stats(eval_stats_class_array, eval_stats_ap_array,
                                              f"checkpoints/best/{model_name}")
                        # ############
                        # ClearML csv reporter logger - V0.3.8
                        # ############
                        if clearml_run:
                            # Report table - CSV from path
                            csv_url = f"{model_logs_path}/{model_name}_eval_stats.csv"
                            task.logger.report_table(
                                "Model evaluation stats",
                                f"{model_name}_eval_stats.csv",
                                iteration=epoch,
                                url=csv_url
                            )

                    ############################
                    # Model evaluation plots logging
                    #############################

                    data = [epoch,
                            args.epochs,
                            precision.mean(),  # Precision
                            recall.mean(),  # Recall
                            AP.mean(),  # mAP
                            f1.mean(),  # f1
                            curr_fitness  # Fitness
                            ]
                    csv_writer(data, model_logs_path + "/" + model_name + "_evaluation_plots.csv", 'a')
                    # ############
                    # ClearML csv reporter logger - V0.3.6
                    # ############
                    if clearml_run:
                        # Report table - CSV from path
                        csv_url = model_logs_path + "/" + model_name + "_evaluation_plots.csv"
                        task.logger.report_table(
                            "Evaluation plots",
                            "evaluation_plots.csv",
                            iteration=epoch,
                            url=csv_url
                        )

            epoch_end = time.time()
            exec_time = epoch_end - epoch_start
            if batches_done >= max_batches:
                print(f'- ❌ - Maximum number of batches reached - {batches_done}/{max_batches} -> Stopping ---- ')
                log_file_writer(f'Maximum number of batches reached - {batches_done}/{max_batches}',
                                "logs/" + date + "_log" + ".txt")
                if args.test_cycle != None:
                    return "Maximum number of batches reached - " + str(batches_done) + "/" + str(max_batches)
                else:
                    exit()
            elif epoch >= args.epochs:
                print(f'- ✅ - Finished training for {args.epochs} epochs ----')
                log_file_writer(f'Finished training for {args.epochs} epochs',
                                model_logfile_path)
                if args.test_cycle is True:
                    return f"Finished training for {args.epochs} epochs, with {req_optimizer} optimizer and {req_scheduler} lr sheduler"
                else:
                    exit()
    except KeyboardInterrupt:
        # Get the current directory
        current_directory = os.getcwd()
        # Define the file path
        file_path = os.path.join(current_directory, "INTERRUPTED.pth")
        if os.path.exists(file_path):
            # Delete the file
            os.remove(file_path)
            print("- ❌ - The old INTERRUPTED.pth has been deleted... -----")
        torch.save(model.state_dict(), f'{model_ckpt_logs_path}/INTERRUPTED.pth')
        print(f'- 💾 - Current weights are saved into {model_ckpt_logs_path}/INTERRUPTED.pth ----')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
    except Exception as e:
        print(f'---- ERROR! -> {traceback.format_exc()}')
        # Create new log file
        f = open("ERROR_log_" + date + ".txt", "w")
        f.close()
        to_print = f"ERROR log - {date} \n Software version: {ver} \n Args: {args} \n Error message: \n {str(traceback.format_exc())}"
        log_file_writer(to_print, "ERROR_log_" + date + ".txt")


if __name__ == "__main__":
    ver = "0.4.8A"
    # Check folders
    check_folders()
    parser = argparse.ArgumentParser(description="Trains the YOLOv3 model.")
    parser.add_argument("-m", "--model", type=str, default="config/yolov3.cfg",
                        help="Path to model definition file (.cfg)")
    parser.add_argument("-d", "--data", type=str, default="config/coco.data",
                        help="Path to data config file (.data)")
    parser.add_argument("--hyp", type=str, default="config/hyp.cfg",
                        help="Path to hyperparameters config file (.cfg)")
    parser.add_argument("-e", "--epochs", type=int, default=300, help="Number of epochs")
    parser.add_argument("-v", "--verbose", action='store_true', help="Makes the training more verbose")
    parser.add_argument("--n_cpu", type=int, default=2, help="Number of cpu threads to use during batch generation")
    parser.add_argument("-pw", "--pretrained_weights", type=str,
                        help="Path to checkpoint file (.weights or .pth). Starts training from checkpoint model")
    parser.add_argument("--evaluation_interval", type=int, default=3,
                        help="Interval of epochs between evaluations on validation set")
    parser.add_argument("--multiscale_training", action="store_true", help="Allow multi-scale training")
    parser.add_argument("--iou_thres", type=float, default=0.1,
                        help="Evaluation: IOU threshold required to qualify as detected")
    parser.add_argument("--conf_thres", type=float, default=0.1, help="Evaluation: Object confidence threshold")
    parser.add_argument("--nms_thres", type=float, default=0.3,
                        help="Evaluation: IOU threshold for non-maximum suppression")
    parser.add_argument("--sync_bn", type=int, default=-1,
                        help="Set use of SyncBatchNorm")
    parser.add_argument("--ema", type=int, default=1,
                        help="Set use of ModelEMA")
    parser.add_argument("--scheduler", type=str, default=None,
                        help="Set type of scheduler")
    parser.add_argument("--optimizer", type=str, default=None,
                        help="Set type of optimizer")
    parser.add_argument("--sampler", type=int, default=0,
                        help="Set type of sampler [0 = None, 1 = SequentialSampler, 2 = RandomSampler, "
                             "3 = SubsetRandomSampler, 4 = WeightedRandomSampler, 5 = BatchSampler]")
    parser.add_argument("--logdir", type=str, default="logs",
                        help="Directory for training log files (e.g. for TensorBoard)")
    parser.add_argument("--name", type=str, default=None,
                        help="Name for trained model")
    parser.add_argument("--warmup", type=bool, default=True,
                        help="Use model training warmup")
    parser.add_argument("--draw", type=bool, default=False,
                        help="Draw evaluation images during training")
    parser.add_argument("--clearml", type=bool, default=False,
                        help="Connect to clearml server")
    parser.add_argument("--test_cycle", type=bool, default=False,
                        help="Define if script should return test feedback")
    parser.add_argument("-g", "--gpu", type=int, default=-1, help="Define which gpu should be used")
    parser.add_argument("--seed", type=int, default=-1, help="Makes results reproducible. Set -1 to disable.")
    args = parser.parse_args()

    print(f"Command line arguments: {args}")
    # Get data configuration
    data_config = parse_data_config(args.data)
    # Get hyperparameters configuration
    hyp_config = parse_hyp_config(args.hyp)
    clearml_cfg = None
    if args.clearml is True:
        # get clearml parameters
        # Create a ConfigParser object
        clearml_cfg = configparser.ConfigParser()
        # Read the config file
        clearml_cfg.read(r'config/clearml.cfg')
    run(args, data_config, hyp_config, ver, clearml_cfg)

# python train.py -m config/yolov3_ITDM_simple.cfg -d config/Nova.data
# -e 10 -v --pretrained_weights weights/yolov3.weights --checkpoint_interval 1 --evaluation_interval 1
