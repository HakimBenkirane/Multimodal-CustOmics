# -*- coding: utf-8 -*-
"""
Created on Wed 01 Sept 2021

@author: Hakim Benkirane

    CentraleSupelec
    MICS laboratory
    9 rue Juliot Curie, Gif-Sur-Yvette, 91190 France

Build the Standard Encoder module.
"""

import torch.nn as nn
from collections import OrderedDict
from src.tools.net_utils import FullyConnectedLayer


class Encoder(nn.Module):
    """
    Neural network Encoder module that creates a feedforward network for encoding input data.

    Args:
    - input_dim (int): Dimensionality of the input data.
    - hidden_dim (list): List of integers representing dimensions of hidden layers.
    - latent_dim (int): Dimensionality of the latent space.
    - norm_layer (nn.Module, optional): Normalization layer (default: nn.BatchNorm1d).
    - leaky_slope (float, optional): Negative slope coefficient for LeakyReLU (default: 0.2).
    - dropout (float, optional): Dropout rate (default: 0).
    - debug (bool, optional): Debug mode flag (default: False).

    Methods:
    - __init__(self, input_dim, hidden_dim, latent_dim, norm_layer=nn.BatchNorm1d, leaky_slope=0.2, dropout=0, debug=False): Initializes the Encoder.
    - forward(self, x): Performs a forward pass through the Encoder.
    - get_outputs(self, x): Retrieves outputs of each layer in the Encoder.
    """

    def __init__(self, input_dim, hidden_dim, latent_dim, norm_layer=nn.BatchNorm1d, leaky_slope=0.2, dropout=0, debug=False):
        """
        Initializes the Encoder module.

        Args:
        - input_dim (int): Dimensionality of the input data.
        - hidden_dim (list): List of integers representing dimensions of hidden layers.
        - latent_dim (int): Dimensionality of the latent space.
        - norm_layer (nn.Module, optional): Normalization layer (default: nn.BatchNorm1d).
        - leaky_slope (float, optional): Negative slope coefficient for LeakyReLU (default: 0.2).
        - dropout (float, optional): Dropout rate (default: 0).
        - debug (bool, optional): Debug mode flag (default: False).
        """
        super(Encoder, self).__init__()

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

        # the output fully-connected layer of the classifier
        self.dt_layers['OutputLayer']= FullyConnectedLayer(hidden_dim[-1], latent_dim, norm_layer=norm_layer, leaky_slope=leaky_slope, dropout=0,
                                 activation=False, normalization=False)

        self.net = nn.Sequential(self.dt_layers)

    def forward(self, x):
        """
        Performs a forward pass through the Encoder.

        Args:
        - x (Tensor): Input data.

        Returns:
        - h (Tensor): Encoded representation.
        """
        h = self.net(x)
        return h

    def get_outputs(self, x):
        """
        Retrieves outputs of each layer in the Encoder.

        Args:
        - x (Tensor): Input data.

        Returns:
        - lt_output (list): List of outputs from each layer.
        """
        lt_output = []
        for layer in self.net:
            lt_output.append(layer(x))
        return lt_output
