"""
A collection of simple plots that I will use during evaulation
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
from tqdm import tqdm
from uloss_wmh.evaluate.eval_metrics.utils import get_n_step_dilated_voxels

def joint_dice_avd_plot(data):
    """
    data: a dictionary or dataframe containing colums dice and avd for some model.
    """
    sns.jointplot(data, x='dice', y='avd')
    plt.xlabel('Dice - per Test Set Individual')
    plt.ylabel('AVD - per Test Set Individual')
    
    
def volume_histogram(volumes):
    plt.hist(volumes, bins=100)
    plt.xlabel(r"WMH Volume per individual - $mm^3$")


def metric_vs_volumes(volumes, df, field, y_label=None):
    if y_label == None:
        y_label = field
    plt.scatter(volumes, df[field])
    plt.hist(volumes, bins=100)
    plt.xlabel(r"WMH Volume per individual - $mm^3$")
    plt.ylabel(y_label)
    
    
def TP_FP_FN_plots(preds3d, ys3d):
    pass

def confidence_dstribution_plot(preds, labels, threshold, dilation_steps):
    prediction_voxels = []
    for i in tqdm(range(len(labels))):
        label = labels[i].cuda()
        pred = preds[i].cuda()
        voxels = get_n_step_dilated_voxels(pred, label, threshold=0.2, steps=2)
        prediction_voxels.append(voxels.reshape(-1).cpu())
    
    prediction_voxels = torch.cat(prediction_voxels)
    sns.histplot(prediction_voxels, bins=20) # seaborn does histplots much much faster than matplotlib...
    plt.xlabel("Softmax Confidence")

    
def calibration_curve_plot(confidences, accuracies):
    plt.plot([0,1],[0,1], c='black') # optimum calibration line
    plt.plot(confidences, accuracies)
    #plt.xlim((0.5, 1))
    #plt.ylim((0., 1))
    plt.xlabel("Mean Softmax Confidence")
    plt.ylabel("Proportion of WMH")
    
    
def pres_recall_thresh_plots(precisions, recalls):
    # skip the first and last bins which go to infinity and 0 respectively
    plt.plot(bins[1:-1], precisions[1:-1], label='precision')
    plt.plot(bins[1:-1], recalls[1:-1], label='recall')
    plt.xlim((0,1))
    plt.ylim((0,1))
    plt.legend(loc='lower right')