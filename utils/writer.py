#################################
# writer.py
# Author: Juha-Matti Rouvinen
# Date: 2023-07-02
# Updated: 2024-01-05
# Version V4.0
##################################
import csv
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
#import matplotlib.pyplot as plt
import numpy as np


def csv_writer(data, filename, oper):
    #header = ['Iterations','Iou Loss','Object Loss','Class Loss','Loss','Learning Rate']
    #header = ['Epoch', 'Epochs', 'Precision', 'Recall', 'mAP', 'F1']
    #log_path = filename.replace("checkpoints", "")
    with open(filename, oper, encoding='UTF8') as f:
        table_writer = csv.writer(f)
        # write the data
        table_writer.writerow(data)
    f.close()

def img_writer_training(iou_loss, obj_loss, cls_loss, loss, lr, batch_loss,iteration, filename):
    #header = ['Epoch', 'Epochs','Iou Loss','Object Loss','Class Loss','Loss','Learning Rate']    # img_writer_data = global_step,x_loss,y_loss,w_loss,h_loss,conf_loss,cls_loss,loss,recall,precision
    #log_path = filename.replace("checkpoints", "")
    # Placing the plots in the plane
    fig = plt.figure(layout="constrained", figsize=(30, 10))
    #fig.set_dpi(1240)
    ax_array = fig.subplots(2, 3, squeeze=False)
    # Using Numpy to create an array x
    x = iteration

    # Plot for iou loss
    ax_array[0, 0].set_ylabel('IoU loss')
    #ax_array[0, 0].plot(x, iou_loss, marker = 'o')
    ax_array[0, 0].plot(x, iou_loss)
    if np.mean(iteration) >= 25:
        #calculate equation for trendline
        z = np.polyfit(x,iou_loss,1)
        p = np.poly1d(z)
        #add trendline plot
        ax_array[0,0].plot(x, p(x))
    #ax_array[0, 0].grid(axis='y', linestyle='-')
    ax_array[0, 0].grid(True)
    ax_array[0, 0].set_xlabel('Iteration')

    # Plot for obj loss
    ax_array[0, 1].set_ylabel('Object loss')
    #ax_array[0, 1].plot(x, obj_loss, marker = 'o')
    ax_array[0, 1].plot(x, obj_loss)
    if np.mean(iteration) >= 25:
        #calculate equation for trendline
        z = np.polyfit(x,obj_loss,1)
        p = np.poly1d(z)
        #add trendline plot
        ax_array[0,1].plot(x, p(x))
    #ax_array[0, 1].grid(axis='y', linestyle='-')
    ax_array[0, 1].grid(True)
    ax_array[0, 1].set_xlabel('Iteration')

    # Plot for cls loss
    ax_array[0, 2].set_ylabel('Class loss')
    #ax_array[0, 2].plot(x, cls_loss, marker = 'o')
    ax_array[0, 2].plot(x, cls_loss)
    if np.mean(iteration) >= 25:
        #calculate equation for trendline
        z = np.polyfit(x,cls_loss,1)
        p = np.poly1d(z)
        #add trendline plot
        ax_array[0,2].plot(x, p(x))
    #ax_array[0, 2].grid(axis='y', linestyle='-')
    ax_array[0, 2].grid(True)
    ax_array[0, 2].set_xlabel('Iteration')

    # Plot for loss
    ax_array[1, 0].set_ylabel('Loss')
    #ax_array[1, 0].plot(x, loss, marker = 'o')
    ax_array[1, 0].plot(x, loss)
    if np.mean(iteration) >= 25:
        #calculate equation for trendline
        z = np.polyfit(x,loss,1)
        p = np.poly1d(z)
        #add trendline plot
        ax_array[1,0].plot(x, p(x))
    #ax_array[1, 0].grid(axis='y', linestyle='-')
    ax_array[1, 0].grid(True)
    ax_array[1, 0].set_xlabel('Iteration')

    # Plot for learning rate
    ax_array[1, 1].set_ylabel('Learning rate')
    #ax_array[1, 1].plot(x, lr, marker = 'o')
    ax_array[1, 1].plot(x, lr)
    # https://stackoverflow.com/questions/21393802/how-to-specify-values-on-y-axis-of-a-matplotlib-plot
    ax_array[1, 1].grid(True)
    ax_array[1, 1].get_autoscaley_on()
    ax_array[1, 1].invert_yaxis()
    if np.mean(iteration) >= 30*(np.min(iteration)+10):
        ax_array[1, 1].set_yscale('log')
    ax_array[1, 1].grid(axis='y', linestyle=' ')
    ax_array[1, 1].set_xlabel('Iteration')

    # Plot for loss
    ax_array[1, 2].set_ylabel('Batch loss')
    #ax_array[1, 2].plot(x, batch_loss, marker='o')
    ax_array[1, 2].plot(x, batch_loss)
    if np.mean(iteration) >= 25:
        #calculate equation for trendline
        z = np.polyfit(x,batch_loss,1)
        p = np.poly1d(z)
        #add trendline plot
        ax_array[1,2].plot(x, p(x))
    #ax_array[1, 2].grid(axis='y', linestyle='-')
    ax_array[1, 2].grid(True)
    ax_array[1, 2].set_xlabel('Iteration')

    fig.savefig(filename+'_training_metrics.png')
    # displaying the title
    plt.title(filename)
    plt.close('all')
    # https://github.com/matplotlib/mplfinance/issues/386 -> failed to allocate bitmap
    fig.clf()

def img_writer_class_dist(weights,classes,values, header,filename):
    fig = plt.figure(layout="constrained", figsize=(20, 10))
    # fig.set_dpi(1240)
    # Using Numpy to create an array x
    ax_array = fig.subplots(1, 2, squeeze=False)

    x = values  # classes
    #y = weights  # class weights

    # Plot for weights
    ax_array[0, 0].set_ylabel('Class weights')
    ax_array[0, 0].bar(x, weights)
    ax_array[0, 0].grid(axis='y', linestyle='-')
    ax_array[0, 0].set_xlabel('Classes')

    # Plot for classes
    ax_array[0, 1].set_ylabel('Class counts')
    ax_array[0, 1].bar(x, classes)
    ax_array[0, 1].grid(axis='y', linestyle='-')
    ax_array[0, 1].set_xlabel('Classes')

    '''
    plt.bar(x, y)
    plt.ylabel('Value')
    plt.xlabel('Classes')
    # displaying the title
    plt.title(header)
    fig.savefig(f'{filename}/{header}_for_dataset.png')
    #plt.close('all')
    '''
    fig.savefig(f'{filename}/{header}_for_dataset.png')
    plt.close('all')
    fig.clf()


def img_writer_evaluation(precision, recall, mAP, f1, ckpt_fitness,train_fitness,epoch, filename):
    #img_writer_evaluation(precision_array, recall_array, mAP_array, f1_array, ap_cls_array, curr_fitness_array, eval_epoch_array, args.logdir + "/" + date)
    # Placing the plots in the plane
    fig = plt.figure(layout="constrained", figsize=(30, 10))
    #fig.set_dpi(1240)
    ax_array = fig.subplots(2, 3, squeeze=False)
    # Using Numpy to create an array x
    x = epoch

    # Plot for precision
    ax_array[0, 0].set_ylabel('Precision')
    ax_array[0, 0].plot(x, precision)
    if np.mean(epoch) >= 25:
        #calculate equation for trendline
        z = np.polyfit(x,precision,1)
        p = np.poly1d(z)
        #add trendline plot
        ax_array[0,0].plot(x, p(x))
    ax_array[0, 0].grid(axis='y', linestyle='-')
    #ax_array[0, 0].invert_yaxis()
    ax_array[0, 0].set_xlabel('Epoch')
    #ax_array[0, 0].set_ybound([0, 1])

    # Plot for recall
    ax_array[0, 1].set_ylabel('Recall')
    ax_array[0, 1].plot(x, recall)
    if np.mean(epoch) >= 25:
        #calculate equation for trendline
        z = np.polyfit(x,recall,1)
        p = np.poly1d(z)
        #add trendline plot
        ax_array[0,1].plot(x, p(x))
    ax_array[0, 1].grid(axis='y', linestyle='-')
    ax_array[0, 1].set_xlabel('Epoch')
    #ax_array[0, 1].set_ybound([0, 1])

    # Plot for f1
    ax_array[0, 2].set_ylabel('F1')
    ax_array[0, 2].plot(x, f1)
    if np.mean(epoch) >= 25:
        #calculate equation for trendline
        z = np.polyfit(x,f1,1)
        p = np.poly1d(z)
        #add trendline plot
        ax_array[0,2].plot(x, p(x))
    ax_array[0, 2].grid(axis='y', linestyle='-')
    ax_array[0, 2].set_xlabel('Epoch')
    # ax_array[1, 0].set_ybound([0, 1])

    # Plot for mAP
    ax_array[1, 0].set_ylabel('mAP')
    ax_array[1, 0].plot(x, mAP)
    if np.mean(epoch) >= 25:
        #calculate equation for trendline
        z = np.polyfit(x,mAP,1)
        p = np.poly1d(z)
        #add trendline plot
        ax_array[1,0].plot(x, p(x))
    ax_array[1, 0].grid(axis='y', linestyle='-')
    ax_array[1, 0].set_xlabel('Epoch')
    #ax_array[0, 2].set_ybound([0, 1])

    # Plot for train fitness
    ax_array[1, 1].set_ylabel('Train FITNESS')
    ax_array[1, 1].plot(x, train_fitness)
    if np.mean(epoch) >= 25:
        #calculate equation for trendline
        z = np.polyfit(x,train_fitness,1)
        p = np.poly1d(z)
        #add trendline plot
        ax_array[1,1].plot(x, p(x))
    ax_array[1, 1].grid(axis='y', linestyle='-')
    ax_array[1, 1].set_xlabel('Epoch')
    #ax_array[1, 1].set_ybound([-1, ])

    # Plot for ckpt fitness
    ax_array[1, 2].set_ylabel('CKPT FITNESS')
    ax_array[1, 2].plot(x, ckpt_fitness)
    if np.mean(epoch) >= 25:
        #calculate equation for trendline
        z = np.polyfit(x,ckpt_fitness,1)
        p = np.poly1d(z)
        #add trendline plot
        ax_array[1,2].plot(x, p(x))
    ax_array[1, 2].grid(axis='y', linestyle='-')
    ax_array[1, 2].set_xlabel('Epoch')
    #ax_array[1, 2].set_ybound([0, 10])

    fig.savefig(filename+'_evaluation_metrics.png')
    plt.close('all')
    fig.clf()


def img_writer_eval_stats(classes,ap,filename):
    #header = ['Epoch', 'Epochs','Iou Loss','Object Loss','Class Loss','Loss','Learning Rate']    # img_writer_data = global_step,x_loss,y_loss,w_loss,h_loss,conf_loss,cls_loss,loss,recall,precision
    #log_path = filename.replace("checkpoints", "")
    # Placing the plots in the plane
    fig = plt.figure(layout="constrained", figsize=(20, 10))
    #fig.set_dpi(1240)
    # Using Numpy to create an array x

    x = classes #classes
    y = ap #AP values

    plt.bar(x, y)
    plt.ylabel('AP')
    plt.xlabel('Classes')

    fig.savefig(filename + '_evaluation_statistics.png')
    # displaying the title
    plt.title(filename)
    plt.close('all')
    fig.clf()


def img_writer_losses(train_loss, eval_loss, epoch, filename):
    #img_writer_evaluation(precision_array, recall_array, mAP_array, f1_array, ap_cls_array, curr_fitness_array, eval_epoch_array, args.logdir + "/" + date)
    # Placing the plots in the plane
    fig = plt.figure(layout="constrained", figsize=(20, 20))
    fig.set_dpi(1240)
    #ax_array = fig.subplots(2, 3, squeeze=False)
    # Using Numpy to create an array x
    x = epoch

    # Plots for losses
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.title('train_eval_losses')
    plt.plot(x, train_loss)
    plt.plot(x, eval_loss)

    fig.savefig(filename + '_train_eval_losses.png')
    plt.close('all')
    fig.clf()


def log_file_writer(data, filename):
    #log_path = filename.replace("checkpoints", "")
    with open(filename, 'a', encoding='UTF8') as f:
        # write the data
        f.write("\n"+data)
    f.close()
