#################################
# optimizer_scheduler_selector.py
# Author: Juha-Matti Rouvinen
# Date: 2024-02-02
# Updated: 2024-02-02
# Version V1
##################################

import math
import torch
from torch import optim
from torch.optim.lr_scheduler import ConstantLR, ExponentialLR
from utils.utils import one_cycle
from utils.writer import log_file_writer


def get_optimizer(req_optimizer, params, pg0, pg1, pg2, lr0, momentum, model,log_file):
    implemented_optimizers = ["adamw", 'sgd', "rmsprop", "adadelta", "adamax", "adam"]
    optimizer = None
    if req_optimizer in implemented_optimizers:
        if req_optimizer == "adamw":
            optimizer = optim.AdamW(
                params,
                lr=float(lr0),
                betas=(float(momentum), 0.999),
                amsgrad=True
            )
        elif req_optimizer == "sgd":
            optimizer = optim.SGD(
                pg0,
                lr=float(lr0),
                momentum=float(momentum),
                nesterov=True,
            )
        elif req_optimizer == "rmsprop":
            optimizer = optim.RMSprop(
                pg0,
                lr=float(lr0),
                momentum=float(momentum)
            )

        elif req_optimizer == "adam":
            optimizer = optim.Adam(
                pg0,
                lr=float(lr0),
                betas=(float(momentum), 0.999),
            )
        elif req_optimizer == "adadelta":
            optimizer = optim.Adadelta(
                pg0,
                lr=float(lr0),
            )
        elif req_optimizer == "adamax":
            optimizer = optim.Adamax(
                pg0,
                lr=float(lr0),
                betas=(float(momentum), 0.999),
            )

    else:
        print("- ⚠ - Unknown optimizer. Reverting into SGD optimizer.")
        log_file_writer(f"- ⚠ - Unknown optimizer. Reverting into SGD optimizer.", log_file)
        optimizer = optim.SGD(
            pg0,
            lr=float(lr0),
            momentum=float(momentum),
            nesterov=True,
        )
        model.hyperparams['optimizer'] = 'sgd'

    return optimizer, model

def get_scheduler(req_scheduler, optimizer,epochs, num_steps, evaluation_interval, dataloader, lr0, lrf, dec_gamma , verbose):
    implemented_schedulers = ['CosineAnnealingLR', 'ChainedScheduler',
                              'ExponentialLR', 'ReduceLROnPlateau', 'ConstantLR',
                              'CyclicLR', 'OneCycleLR', 'LambdaLR', 'MultiplicativeLR',
                              'StepLR', 'MultiStepLR', 'LinearLR', 'PolynomialLR', 'CosineAnnealingWarmRestarts']

    scheduler = None
    if req_scheduler in implemented_schedulers:
        lf = lambda x: ((1 + math.cos(x * math.pi / epochs)) / 2) * (1 - float(lrf)) + float(
            lrf)  # cosine

        # CosineAnnealingLR
        if req_scheduler == 'CosineAnnealingLR':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=int(num_steps / 10),
                eta_min=float(lr0) / 10000,
                verbose=False)
        # ChainedScheduler
        elif req_scheduler == 'ChainedScheduler':
            scheduler1 = ConstantLR(optimizer, factor=0.5, total_iters=5,
                                    verbose=False)
            scheduler2 = ExponentialLR(optimizer, gamma=float(dec_gamma), verbose=False)
            scheduler = torch.optim.lr_scheduler.ChainedScheduler([scheduler1, scheduler2])
        elif req_scheduler == 'ExponentialLR':
            scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=float(dec_gamma),
                                                               verbose=False)
        elif req_scheduler == 'ReduceLROnPlateau':
            minimum_lr = float(lr0) * float(lrf)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                'min',
                patience=int(evaluation_interval) * 10,
                min_lr=minimum_lr,
                cooldown=int(evaluation_interval/2),
                verbose=False)
        elif req_scheduler == 'ConstantLR':
            scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer, factor=0.5,
                                                            total_iters=int(evaluation_interval),
                                                            verbose=False)
        elif req_scheduler == 'CyclicLR':
            scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer,
                                                          base_lr=float(lr0) * float(
                                                              lrf),
                                                          max_lr=float(lr0), cycle_momentum=True,
                                                          verbose=False,
                                                          mode='exp_range')  # mode (str): One of {triangular, triangular2, exp_range}.
        elif req_scheduler == 'OneCycleLR':
            scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=float(lrf),
                                                            steps_per_epoch=len(dataloader),
                                                            epochs=int(epochs))
        elif req_scheduler == 'LambdaLR':
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,
                                                          lr_lambda=lf,
                                                          verbose=False)  # plot_lr_scheduler(optimizer, scheduler, epochs)
        elif req_scheduler == 'MultiplicativeLR':
            lf = one_cycle(1, float(lr0), epochs)  # cosine 1->hyp['lrf']
            scheduler = torch.optim.lr_scheduler.MultiplicativeLR(optimizer, lr_lambda=lf)
        elif req_scheduler == 'StepLR':
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=int(evaluation_interval),
                                                        gamma=0.1)  # Step size -> epochs
        elif req_scheduler == 'MultiStepLR':
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30, 80],
                                                             gamma=0.1)  # milestones size -> epochs
        elif req_scheduler == 'LinearLR':
            scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.5, total_iters=int(epochs/2))  # total_iters size -> epochs
        elif req_scheduler == 'PolynomialLR':
            scheduler = torch.optim.lr_scheduler.PolynomialLR(optimizer, total_iters=int(epochs),
                                                              power=1.0)  # total_iters size -> epochs
        elif req_scheduler == 'CosineAnnealingWarmRestarts':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=int(num_steps / 10),
                                                                             eta_min=float(
                                                                                 lr0) * float(lrf))  # total_iters size -> epochs
    else:
        print("- ⚠ - Unknown scheduler! Reverting to LambdaLR")
        req_scheduler = 'LambdaLR'
        lf = lambda x: (1 - x / epochs) * (1.0 - float(lrf)) + float(lrf)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,
                                                      lr_lambda=lf, verbose=False)

    return scheduler