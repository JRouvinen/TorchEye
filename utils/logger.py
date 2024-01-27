#################################
# Logger.py
# Author: Juha-Matti Rouvinen
# Date: 2023-07-02
# Updated: 2024-01-27
# Version V3.0
##################################
import os
import datetime

import torchvision.utils
from torch.utils.tensorboard import SummaryWriter


class Logger(object):
    def __init__(self, log_dir, log_hist=True):
        """Create a summary writer logging to log_dir."""
        if log_hist:    # Check a new folder for each log should be dreated
            log_dir = os.path.join(
                log_dir,
                datetime.datetime.now().strftime("%Y_%m_%d__%H_%M_%S"))
        self.writer = SummaryWriter(log_dir)

    def scalar_summary(self, tag, value, step):
        """Log a scalar variable."""
        self.writer.add_scalar(tag, value, step)

    def list_of_scalars_summary(self, tag_value_pairs, step):
        """Log scalar variables."""
        for tag, value in tag_value_pairs:
            self.writer.add_scalar(tag, value, step)

    def add_image(self,name,img):
        """Log images - V2"""
        self.writer.add_image(name, img)

    def add_pr_curve(self,name, labels, predictions, epoch):
        self.writer.add_pr_curve('pr_curve',labels,predictions,epoch)

    def add_graph(self,model,images):
        self.writer.add_graph(model, images)

    def add_figure(self,tag,fig,global_step,close=True,walltime=None):
        self.writer.add_figure(tag,fig,global_step,close,walltime)