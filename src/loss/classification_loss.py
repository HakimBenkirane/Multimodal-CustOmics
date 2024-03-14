# -*- coding: utf-8 -*-
"""
Created on Wed 01 Sept 2021

@author: Hakim Benkirane

    CentraleSupelec
    MICS laboratory
    9 rue Juliot Curie, Gif-Sur-Yvette, 91190 France

Conception of the classification loss.
"""

import torch.nn as nn



def classification_loss(loss_name, y_true, y_pred ,reduction='mean'):
    """
    Computes the classification loss based on the specified loss function.

    Args:
    - loss_name (str): Name of the loss function ('BCE' for Binary Cross-Entropy, 'CE' for Cross Entropy).
    - y_true (tensor): True labels.
    - y_pred (tensor): Predicted scores/logits.
    - reduction (str, optional): Specifies the reduction to apply to the loss. Default is 'mean'.

    Returns:
    - tensor: Calculated classification loss based on the specified loss function.
    
    Raises:
    - NotImplementedError: If the provided loss function is not supported.
    """
    if loss_name == 'BCE':
        return nn.BCEWithLogitsLoss(reduction=reduction)(y_true, y_pred)
    elif loss_name == 'CE':
        return nn.CrossEntropyLoss(reduction=reduction)(y_true, y_pred)
    else:
        raise NotImplementedError('Loss function %s is not found' % loss_name)