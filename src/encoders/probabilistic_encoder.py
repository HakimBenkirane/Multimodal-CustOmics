# -*- coding: utf-8 -*-
"""
Created on Wed 01 Sept 2021

@author: Hakim Benkirane

    CentraleSupelec
    MICS laboratory
    9 rue Juliot Curie, Gif-Sur-Yvette, 91190 France

Build the Probabilistic Encoder module.
"""

import torch.nn as nn
from collections import OrderedDict
from src.tools.net_utils import FullyConnectedLayer

class ProbabilisticEncoder(nn.Module):
    """
    Neural network Probabilistic Encoder module that creates a feedforward network for probabilistic encoding of input data.

    Args:
    - input_dim (int): Dimensionality of the input data.
    - hidden_dim (list): List of integers representing dimensions of hidden layers.
    - latent_dim (int): Dimensionality of the latent space.
    - norm_layer (nn.Module, optional): Normalization layer (default: nn.BatchNorm1d).
    - leaky_slope (float, optional): Negative slope coefficient for LeakyReLU (default: 0.2).
    - dropout (float, optional): Dropout rate (default: 0).
    - debug (bool, optional): Debug mode flag (default: False).

    Methods:
    - __init__(self, input_dim, hidden_dim, latent_dim, norm_layer=nn.BatchNorm1d, leaky_slope=0.2, dropout=0, debug=False): Initializes the ProbabilisticEncoder.
    - forward(self, x): Performs a forward pass through the ProbabilisticEncoder.
    """

    def __init__(self, input_dim, hidden_dim, latent_dim, norm_layer=nn.BatchNorm1d, leaky_slope=0.2, dropout=0, debug=False):
        """
        Initializes the ProbabilisticEncoder module.

        Args:
        - input_dim (int): Dimensionality of the input data.
        - hidden_dim (list): List of integers representing dimensions of hidden layers.
        - latent_dim (int): Dimensionality of the latent space.
        - norm_layer (nn.Module, optional): Normalization layer (default: nn.BatchNorm1d).
        - leaky_slope (float, optional): Negative slope coefficient for LeakyReLU (default: 0.2).
        - dropout (float, optional): Dropout rate (default: 0).
        - debug (bool, optional): Debug mode flag (default: False).
        """
        super(ProbabilisticEncoder, self).__init__()

        self.dt_layers = OrderedDict()

        self.dt_layers['InputLayer'] = FullyConnectedLayer(input_dim, hidden_dim[0], norm_layer=norm_layer, leaky_slope=leaky_slope, dropout=dropout,
                                activation=True)

        block_layer_num = len(hidden_dim)
        dropout_flag = True
        for num in range(1, block_layer_num):
            self.dt_layers['Layer{}'.format(num)] = FullyConnectedLayer(hidden_dim[num - 1], hidden_dim[num], norm_layer=norm_layer, leaky_slope=leaky_slope,
                                    dropout=dropout_flag*dropout, activation=True)
            # dropout for every other layer
            dropout_flag = not dropout_flag

        self.net = nn.Sequential(self.dt_layers)

        # The output fully-connected layers for mean and log variance
        self.mean_layer = FullyConnectedLayer(hidden_dim[-1], latent_dim, norm_layer=norm_layer, leaky_slope=leaky_slope, dropout=0,
                                 activation=False, normalization=False)
        self.log_var_layer = FullyConnectedLayer(hidden_dim[-1], latent_dim, norm_layer=norm_layer, leaky_slope=leaky_slope, dropout=0,
                                 activation=False, normalization=False)

    def forward(self, x):
        """
        Performs a forward pass through the ProbabilisticEncoder.

        Args:
        - x (Tensor): Input data.

        Returns:
        - mean (Tensor): Output representing mean of the encoded distribution.
        - log_var (Tensor): Output representing logarithm of variance of the encoded distribution.
        """
        h = self.net(x)
        mean = self.mean_layer(h)
        log_var = self.log_var_layer(h)
        
        return mean, log_var






