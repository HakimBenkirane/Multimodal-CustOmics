# -*- coding: utf-8 -*-
"""
Created on Wed 01 Sept 2021

@author: Hakim Benkirane

    CentraleSupelec
    MICS laboratory
    9 rue Juliot Curie, Gif-Sur-Yvette, 91190 France

Creates the different metrics to evaluate a classification task.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import metrics
from sklearn.metrics import roc_auc_score, roc_curve, auc
from sklearn.preprocessing import OneHotEncoder

def roc_auc_score_multiclass(y_true, y_pred, ohe, average = "macro"):
    """
    Computes the ROC AUC score for multiclass classification.

    Args:
    - y_true (array-like): True labels.
    - y_pred (array-like): Predicted labels.
    - ohe (object): OneHotEncoder object.
    - average (str): Method for calculating scores for multiclass. Default is "macro".

    Returns:
    - float: ROC AUC score.
    """
    #y_true = ohe.transform(np.array(y_true).reshape(-1,1))

    roc_auc = roc_auc_score(y_true, y_pred, average = average, multi_class='ovo')

    return roc_auc

def multi_classification_evaluation(y_true, y_pred, y_pred_proba, average='weighted', save_confusion=False, filename=None, plot_roc=False, ohe=None):
    """
    Computes various classification metrics and optionally saves a confusion matrix plot or ROC curves plot.

    Args:
    - y_true (array-like): True labels.
    - y_pred (array-like): Predicted labels.
    - y_pred_proba (array-like): Predicted probabilities.
    - average (str): Method for calculating scores. Default is "weighted".
    - save_confusion (bool): Flag to save confusion matrix plot. Default is False.
    - filename (str): File name for the saved plot.
    - plot_roc (bool): Flag to plot ROC curves. Default is False.
    - ohe (object): OneHotEncoder object.

    Returns:
    - dict: Dictionary containing computed metrics.
    """

    accuracy = metrics.accuracy_score(y_true, y_pred)
    precision = metrics.precision_score(y_true, y_pred, average=average)
    recall = metrics.recall_score(y_true, y_pred, average = average)
    f1_score = metrics.f1_score(y_true, y_pred, average = average)
    auc = roc_auc_score_multiclass(y_true, y_pred_proba, ohe, average = average)
    dt_scores = {'Accuracy': accuracy,
                'F1-score' : f1_score,
                'Precision' : precision,
                'Recall' : recall,
                'AUC' : auc} 

    if save_confusion:
        plt.figure(figsize = (18,8))
        sns.heatmap(metrics.confusion_matrix(y_true, y_pred), annot = True, xticklabels = np.unique(y_true), yticklabels = np.unique(y_true), cmap = 'summer')
        plt.xlabel('Predicted Labels')
        plt.ylabel('True Labels')
        plt.savefig(filename + '.png')
        plt.clf()

    return dt_scores


def plot_roc_multiclass(y_test, y_pred_proba, filename="", n_classes=2, var_names=['CMML', 'MDS'], dmat=False):
    """
    Plots ROC curves for multiclass classification.

    Args:
    - y_test (array-like): True labels.
    - y_pred_proba (array-like): Predicted probabilities.
    - filename (str): File name for the saved plot. Default is an empty string.
    - n_classes (int): Number of classes. Default is 2.
    - var_names (list): List of class names. Default is ["CMML", "MDS"].
    - dmat (bool): Flag for dynamic matrix. Default is False.
    """
    y_score = y_pred_proba

    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        y_binarized = (y_test == i)
        y_scores_i = y_score[:,i]
        fpr[i], tpr[i], _ = roc_curve(y_binarized, y_scores_i)
        roc_auc[i] = auc(fpr[i], tpr[i])


    # Plot all ROC curves
    plt.figure()
    random_color = lambda : [np.random.rand() for _ in range(3)]
    #colors = [random_color() for _ in range(n_classes)]
    colors = ["red", "green", "blue", "magenta"]
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=1,
                 label='class {0} (AUC = {1:0.2f})'
                 ''.format(var_names[i], roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=2, label="random")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curve for multi-class model {}'.format(filename))
    plt.legend(loc="lower right")
    if (filename != ""): plt.savefig("roc_multi_{}.png".format(filename))
    #else: #plt.show()



    