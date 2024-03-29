# -*- coding: utf-8 -*-
"""
Created on Wed 01 Sept 2021

@author: Hakim Benkirane

    CentraleSupelec
    MICS laboratory
    9 rue Juliot Curie, Gif-Sur-Yvette, 91190 France

Conception of the mean-maximum discrepency loss.
"""

import torch



def compute_kernel(x, y):
    """
    Computes the kernel matrix between two sets of vectors x and y.

    Args:
    - x (tensor): Input tensor x of shape (x_size, dim).
    - y (tensor): Input tensor y of shape (y_size, dim).

    Returns:
    - tensor: Computed kernel matrix of shape (x_size, y_size).
    """

    x_size = x.size(0)
    y_size = y.size(0)
    dim = x.size(1)
    x = x.unsqueeze(1) # (x_size, 1, dim)
    y = y.unsqueeze(0) # (1, y_size, dim)
    tiled_x = x.expand(x_size, y_size, dim)
    tiled_y = y.expand(x_size, y_size, dim)
    kernel_input = (tiled_x - tiled_y).pow(2).mean(2)/float(dim)
    return torch.exp(-kernel_input) # (x_size, y_size)

def compute_mmd(x, y):
    """
    Computes the Maximum Mean Discrepancy (MMD) between two sets of vectors x and y.

    Args:
    - x (tensor): Input tensor x of shape (x_size, dim).
    - y (tensor): Input tensor y of shape (y_size, dim).

    Returns:
    - float: Computed Maximum Mean Discrepancy (MMD) value.
    """

    x_kernel = compute_kernel(x, x)
    y_kernel = compute_kernel(y, y)
    xy_kernel = compute_kernel(x, y)
    mmd = x_kernel.mean() + y_kernel.mean() - 2*xy_kernel.mean()
    return mmd
