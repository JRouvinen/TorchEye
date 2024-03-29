#################################
# writer.py
# Author: Juha-Matti Rouvinen
# Date: 2023-07-02
# Updated: 2024-02-09
# Version V5.2
##################################
import csv
import matplotlib
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.pyplot import hist2d
#from utils import threaded
from utils.loss import fitness
from matplotlib import pyplot as plt
matplotlib.use("Agg")

def open_file(path):
    file = matplotlib.pyplot.imread(path, format=None)
    return file
#@threaded
def csv_writer(data, filename, oper):
    #header = ['Iterations','Iou Loss','Object Loss','Class Loss','Loss','Learning Rate']
    #header = ['Epoch', 'Epochs', 'Precision', 'Recall', 'mAP', 'F1']
    #log_path = filename.replace("checkpoints", "")
    with open(filename, oper, encoding='UTF8') as f:
        table_writer = csv.writer(f)
        # write the data
        table_writer.writerow(data)
    f.close()

def csvDictWriter(data, filename, header):
    # my data rows as dictionary objects
    mydict = data

    # field names
    fields = header

    # name of csv file
    filename = filename

    # writing to csv file
    with open(filename, 'w') as csvfile:
        # creating a csv dict writer object
        writer = csv.DictWriter(csvfile, fieldnames=fields)

        # writing headers (field names)
        writer.writeheader()

        # writing data rows
        writer.writerows(mydict)

#@threaded
def img_writer_training(iou_loss, obj_loss, cls_loss, loss, lr, batch_loss,iteration, filename,logger):
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
    if np.mean(iteration) >= 100:
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
    if np.mean(iteration) >= 100:
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
    if np.mean(iteration) >= 100:
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
    if np.mean(iteration) >= 100:
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
    if np.mean(iteration) >= 100:
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
    if logger is not None:
        logger.add_figure('training_metrics', fig, global_step=iteration.max(), close=True, walltime=None)
    plt.close('all')
    # https://github.com/matplotlib/mplfinance/issues/386 -> failed to allocate bitmap
    fig.clf()
#@threaded
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
    #if logger is not None:
    #    logger.add_figure(f'{header}_for_dataset', fig, global_step=1, close=True, walltime=None)
    plt.close('all')
    fig.clf()
#@threaded
def img_writer_evaluation(precision, recall, mAP, f1, ckpt_fitness,train_fitness,epoch, filename,logger):
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
    if logger is not None:
        logger.add_figure(f'evaluation_metrics', fig, global_step=epoch.max(), close=True, walltime=None)
    plt.close('all')
    fig.clf()

#@threaded
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

#@threaded
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

#@threaded
def plot_evolve(evolve_csv="path/to/evolve.csv"):  # from utils.plots import *; plot_evolve()
    # Plot evolve.csv hyp evolution results
    evolve_csv = evolve_csv
    data = pd.read_csv(evolve_csv)
    keys = [x.strip() for x in data.columns]
    x = data.values
    f = fitness(x)
    j = np.argmax(f)  # max fitness index
    plt.figure(figsize=(10, 12), tight_layout=True)
    matplotlib.rc("font", **{"size": 8})
    print(f"Best results from row {j} of {evolve_csv}:")
    for i, k in enumerate(keys[7:]):
        v = x[:, 7 + i]
        mu = v[j]  # best single result
        plt.subplot(6, 5, i + 1)
        plt.scatter(v, f, c=hist2d(v, f, 20), cmap="viridis", alpha=0.8, edgecolors="none")
        plt.plot(mu, f.max(), "k+", markersize=15)
        plt.title(f"{k} = {mu:.3g}", fontdict={"size": 9})  # limit to 40 characters
        if i % 5 != 0:
            plt.yticks([])
        print(f"{k:>15}: {mu:.3g}")
    f = evolve_csv.with_suffix(".png")  # filename
    plt.savefig(f, dpi=200)
    plt.close()
    print(f"Saved {f}")
#@threaded
def log_file_writer(data, filename):
    #log_path = filename.replace("checkpoints", "")
    with open(filename, 'a', encoding='UTF8') as f:
        # write the data
        f.write("\n"+data)
    f.close()

def img_polar_chart(ap_data, log_path, names):
    '''
    Generate polar_chart for auc scores.
    auc_scores : [dict] auc_scores
    names : [list] cls names
    return None
    save img at Path(save_dir) / 'polar_chart.png'
    '''

    # Creating a new figure and setting up the resolution
    fig = plt.figure(dpi=200)

    # Change the coordinate system from scaler to polar
    ax = fig.add_subplot(projection='polar')

    # Generating the X and Y axis data points
    #r = [8, 8, 8, 8, 8, 8, 8, 8, 8]
    r =  1
    indx = 0
    values = []
    # creating an array containing the
    # radian values
    for i in ap_data:
        if i[0] == indx:
            values.append(i[2])
        else:
            values.append(0)
        indx += 1

    #rads = np.arange(0, values, 1)
    # plotting the circle
    for rad in values:
        plt.polar(rad, r, 'g.')

    fig.savefig(log_path + '_polar_plot.png')
    # displaying the title
    plt.title('polar_plot')
    plt.close('all')
    fig.clf()
