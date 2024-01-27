#################################
# auc_roc.py
# Author: Juha-Matti Rouvinen
# Date: 2024-01-24
# Updated: 2024-01-27
# Version V1.2
# This implementation is based on the code from: https://github.com/haooyuee/YOLOv5-AUC-ROC-MedDetect/
##################################

#Imports
import torch
#from torchvision.ops import box_iou
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from pathlib import Path
from scipy.signal import savgol_filter
from scipy.interpolate import make_interp_spline

from utils.utils import box_iou, xywh2xyxy


class AUROC:
    """ Compute the auroc scores, given the auc for each class.
    """

    def __init__(self, nc, conf=0.25, iou_thres=0.45):
        """
        different thresholds (like prediction confidence, Iou) will have a great impact on the results of AUC.
        Default: conf=0.25, iou_thres=0.45
        """
        self.auc_scores = np.zeros(nc)  # Store the AUROC score for each cls
        self.nc = nc  # number of cls
        self.conf = conf  # confidence threshold
        self.iou_thres = iou_thres  # IoU threshold

        self.pred = [[] for _ in range(nc)]  # list to store model predictions for each class
        self.true = [[] for _ in range(nc)]  # list to store ground truth labels for each class

        #import subprocess
        #subprocess.check_call(['pip', 'install', 'scikit-learn'])
        #subprocess.check_call(['pip', 'install', 'plotly', 'kaleido'])

    def generate_batch_data(self, outputs, targets):
        for si, pred in enumerate(outputs):
            out_labels = targets[targets[:, 0] == si, 1:]
            nl, npr = out_labels.shape[0], pred.shape[0]  # number of labels, predictions
            predn = pred.clone()
            # Evaluate
            if nl:
                tbox = xywh2xyxy(out_labels[:, 1:5])  # target boxes
                #scale_boxes(_[si].shape[1:], tbox, shape, shapes[si][1])  # native-space labels
                labelsn = torch.cat((out_labels[:, 0:1], tbox), 1)  # native-space labels
                #correct = self.process_batch(predn, labelsn, iouv)
                self.process_batch(predn, out_labels)

    def process_batch(self, detections, labels):
        """
        Return /
        Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
        Arguments:
            detections (Array[N, 6]), x1, y1, x2, y2, conf, class
            labels (Array[M, 5]), class, x1, y1, x2, y2
        Returns:
            None, updates pred[list] and true[list] accordingly
        """
        if detections is None:
            # If there is no prediction result, all ground truths are considered to be negative samples,
            # Ignored during calculating auc
            return

        t = 0

        detections = detections[detections[:, 4] > self.conf]
        # Filter out prediction db boxes with low confidence (similar to nms)
        gt_classes = labels[:, 0].int()  # All gt box categories (int) cls, may be repeated
        detection_classes = detections[:, 5].int()  # All pred box cls (int) categories, may repeat positive + negative
        iou = box_iou(labels[:, 1:], detections[:, :4])  # # Find the iou of all gt boxes and all pred boxes

        x = torch.where(iou > self.iou_thres)  # Filtered by iou threshold

        if x[0].shape[0]:  # When have iou > iou threshold
            matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1).cpu().numpy()
            # cat gt_index+pred_index+iou
            if x[0].shape[0] > 1:
                matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
                # finally get the one with the largest iou of each db pred and all db gt (greater than the iou_thres)
                # Each db gt will only correspond to the only one db pred. The filtered preds are all positive samples _> (tp or fp)
        else:
            matches = np.zeros((0, 3))

        n = matches.shape[0] > 0
        m0, m1, _ = matches.transpose().astype(int)
        for class_id in range(self.nc):
            for i, gc in enumerate(gt_classes):
                if gc == class_id:
                    j = m0 == i
                    if n and sum(j) == 1:
                        if detection_classes[m1[j]] == gc:
                            # same cls -> True Positive
                            self.pred[class_id].append(detections[m1[j], 4].item())  # save conf
                            self.true[class_id].append(1)  # True Positive set 1
                        else:
                            # diff cls -> False Positive
                            self.pred[class_id].append(detections[m1[j], 4].item())  # save conf
                            self.true[class_id].append(0)  # False Positive set 0
                        t = t + 1
                    '''
                    # Ignored during calculating auc
                    else:
                        self.pred[class_id].append(0.0)
                        self.true[class_id].append(-1)
                    '''

    def out(self):
        '''
        Computes the AUROC score for each category and returns it.
        '''
        from sklearn.metrics import roc_auc_score, roc_curve
        auc_scores = np.zeros(self.nc)
        fpr_ = [[] for _ in range(self.nc)]
        tpr_ = [[] for _ in range(self.nc)]

        for class_id in range(self.nc):
            labels = self.true[class_id]
            preds = self.pred[class_id]
            try:
                fpr_class, tpr_class, _ = roc_curve(labels, preds)
                auc_scores[class_id] = roc_auc_score(labels, preds)
                fpr_[class_id] = fpr_class
                tpr_[class_id] = tpr_class

            except ValueError:
                # No pred = set auc to 0
                # print('No pred db for cls ' + str(class_id) + ', Set the auc value to 0 ...')
                auc_scores[class_id] = 0

        return auc_scores, fpr_, tpr_

    def plot_polar_chart(self, auc_scores, save_dir='', names=()):
        '''
        Generate polar_chart for auc scores.
        auc_scores : [dict] auc_scores
        names : [list] cls names
        return None
        save img at Path(save_dir) / 'polar_chart.png'
        '''
        mauc = auc_scores.mean()
        auc_scores_name = dict(zip(names, auc_scores))
        auc_scores_name['mAUC'] = mauc
        df = pd.DataFrame.from_dict(auc_scores_name, orient='index')
        columns = list(df.index)
        fig = go.Figure(
            data=[go.Scatterpolar(r=(df[0] * 100).round(0), fill='toself', name='Classes', theta=columns)],
            layout=go.Layout(
                # title=go.layout.Title(text='Class AUC'),
                polar={
                    'radialaxis': {
                        'range': [0, 100],
                        'tickvals': [0, 25, 50, 75, 100],
                        'ticktext': ['0%', '25%', '50%', '75%', '100%'],
                        'visible': True, }},
                showlegend=True,
                template='plotly_dark',
            ),
        )
        file_name = Path(save_dir) / 'polar_chart.png'
        fig.write_image(file_name)
        plt.close('all')
        #print('plot_polar_chart DONE')

    def plot_auroc_curve(self, fpr_, tpr_, auc_scores, save_dir='', names=(),epoch=0,logger=None):
        # AUROC curve
        fig, ax = plt.subplots(1, 1, figsize=(9, 6), tight_layout=True)
        if 0 < len(names) < 21:  # display per-class legend if < 21 classes
            #ax.plot(fpr_[1], tpr_[1], linewidth=1, label=f'{names} {auc_scores}')
            #X_Y_Spline = make_interp_spline(tpr_[1], fpr_[1])
            # Returns evenly spaced numbers
            # over a specified interval.
            x_tensor = tpr_[1]
            X_ = np.linspace(x_tensor.min(), x_tensor.max(),x_tensor.size)
            Y_ = fpr_[1]
            yhat = savgol_filter(Y_, Y_.size, 3)  # window size 51, polynomial order 3
            # Plotting the Graph
            ax.plot(X_, yhat, linewidth=1, color='blue')
            # Generate label-score dict
            #label_score_dict = {}
            indx = 0
            textstr = ""
            for x in names.values():
                #label_score_dict[x] = f'{auc_scores[indx]:.3f}'
                textstr += f'\n{x} - {auc_scores[indx]:.3f}'
                indx += 1

            # these are matplotlib.patch.Patch properties
            props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
            # place a text box in upper left in axes coords
            ax.text(0.8, 0.1, textstr, transform=ax.transAxes, fontsize=10,
                    verticalalignment='bottom', bbox=props)

        else:
            x_tensor = tpr_[1]
            X_ = np.linspace(x_tensor.min(), x_tensor.max(), x_tensor.size)
            Y_ = fpr_[1]
            yhat = savgol_filter(Y_, Y_.size, 3)  # window size 51, polynomial order 3
            # Plotting the Graph
            ax.plot(X_, yhat, linewidth=1, color='blue')

        ax.plot([0, 1], [0, 1], linestyle='--', color='black', linewidth=1)  # diagonal line
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        #ax.legend(bbox_to_anchor=(1.04, 1), loc='upper left')
        ax.set_title(f'AUROC Curve - Epoch {epoch}')
        if save_dir:
            save_path = Path(save_dir) / 'auc_roc_curve_last.png'
            fig.savefig(save_path, dpi=250)
            # print(f'Saved AUROC curve at: {save_path}')
        if logger is not None:
            logger.add_figure('auc_roc_curve', fig, global_step=epoch, close=True,
                              walltime=None)
        plt.close(fig)
        plt.close('all')
        fig.clf()
        #print('plot_auroc_curve DONE')
