# -*- coding: utf-8 -*-
"""
Created on Wed 01 Sept 2021

@author: Hakim Benkirane

    CentraleSupelec
    MICS laboratory
    9 rue Juliot Curie, Gif-Sur-Yvette, 91190 France

Build the Standard Decoder module.
"""

import torch
import torch.nn as nn
from collections import OrderedDict
from src.tools.net_utils import FullyConnectedLayer


class Decoder(nn.Module):
    """
    Neural network Decoder module that creates a feedforward network for decoding latent representations.

    Args:
    - latent_dim (int): Dimensionality of the latent space.
    - hidden_dim (list): List of integers representing dimensions of hidden layers.
    - output_dim (int): Dimensionality of the output data.
    - norm_layer (nn.Module, optional): Normalization layer (default: nn.BatchNorm1d).
    - leaky_slope (float, optional): Negative slope coefficient for LeakyReLU (default: 0.2).
    - dropout (float, optional): Dropout rate (default: 0).

    Methods:
    - __init__(self, latent_dim, hidden_dim, output_dim, norm_layer=nn.BatchNorm1d, leaky_slope=0.2 ,dropout=0): Initializes the Decoder.
    - forward(self, x): Performs a forward pass through the Decoder.
    """

    def __init__(self, latent_dim, hidden_dim, output_dim, norm_layer=nn.BatchNorm1d, leaky_slope=0.2, dropout=0):
        """
        Initializes the Decoder module.

        Args:
        - latent_dim (int): Dimensionality of the latent space.
        - hidden_dim (list): List of integers representing dimensions of hidden layers.
        - output_dim (int): Dimensionality of the output data.
        - norm_layer (nn.Module, optional): Normalization layer (default: nn.BatchNorm1d).
        - leaky_slope (float, optional): Negative slope coefficient for LeakyReLU (default: 0.2).
        - dropout (float, optional): Dropout rate (default: 0).
        """
        super(Decoder, self).__init__()

        self.dt_layers = OrderedDict()

        self.dt_layers['InputLayer'] = FullyConnectedLayer(latent_dim, hidden_dim[0], norm_layer=norm_layer, leaky_slope=leaky_slope, dropout=dropout,
                                activation=True)

        block_layer_num = len(hidden_dim)
        dropout_flag = True
        for num in range(1, block_layer_num):
            self.dt_layers['Layer{}'.format(num)] = FullyConnectedLayer(hidden_dim[num - 1], hidden_dim[num], norm_layer=norm_layer, leaky_slope=leaky_slope,
                                    dropout=dropout_flag*dropout, activation=True)
            # dropout for every other layer
            dropout_flag = not dropout_flag

        self.dt_layers['OutputLayer'] = FullyConnectedLayer(hidden_dim[-1], output_dim, norm_layer=norm_layer, leaky_slope=leaky_slope, dropout=0,
                                 activation=False, normalization=False)

        self.net = nn.Sequential(self.dt_layers)

    def forward(self, x):
        """
        Performs a forward pass through the Decoder.

        Args:
        - x (Tensor): Input data.

        Returns:
        - x_hat (Tensor): Decoded output data.
        """
        x_hat = self.net(x)
        return x_hat
