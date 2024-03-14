# -*- coding: utf-8 -*-
"""
Created on Wed 01 Sept 2021

@author: Hakim Benkirane

    CentraleSupelec
    MICS laboratory
    9 rue Juliot Curie, Gif-Sur-Yvette, 91190 France

Sets-up the different types of layers that can be used.
"""

import torch.nn as nn


class FullyConnectedLayer(nn.Module):
    """
    Implements a fully connected layer with optional normalization, dropout, and activation functions.

    Args:
    - input_dim (int): Input dimension of the layer.
    - output_dim (int): Output dimension of the layer.
    - norm_layer (nn.Module, optional): Normalization layer (default: nn.BatchNorm1d).
    - leaky_slope (float, optional): Negative slope for LeakyReLU activation (default: 0.2).
    - dropout (float, optional): Dropout rate (default: 0.2).
    - activation (bool, optional): Whether to apply activation function (default: True).
    - normalization (bool, optional): Whether to apply normalization (default: True).
    - activation_name (str, optional): Name of the activation function (default: 'LeakyReLU').

    Methods:
    - forward(x): Forward pass through the layer.

    Note:
    - Supported activation functions: 'ReLU', 'Sigmoid', 'LeakyReLU', 'Tanh', 'Softmax', 'No'.
    """

    def __init__(self, input_dim, output_dim, norm_layer=nn.BatchNorm1d, leaky_slope=0.2, dropout=0.2,
                 activation=True, normalization=True, activation_name='LeakyReLU'):
        super(FullyConnectedLayer, self).__init__()
        # Linear
        self.fc_block = [nn.Linear(input_dim, output_dim)]
        # Norm
        if normalization:
            norm_layer = nn.BatchNorm1d
            self.fc_block.append(norm_layer(output_dim))
        # Dropout
        if 0 < dropout <= 1:
            self.fc_block.append(nn.Dropout(p=dropout))
        # LeakyReLU
        if activation:
            if activation_name.lower() == 'relu':
                self.fc_block.append(nn.ReLU())
            elif activation_name.lower() == 'sigmoid':
                self.fc_block.append(nn.Sigmoid())
            elif activation_name.lower() == 'leakyrelu':
                self.fc_block.append(nn.LeakyReLU(negative_slope=leaky_slope, inplace=True))
            elif activation_name.lower() == 'tanh':
                self.fc_block.append(nn.Tanh())
            elif activation_name.lower() == 'softmax':
                self.fc_block.append(nn.Softmax(dim=1))
            elif activation_name.lower() == 'no':
                pass
            else:
                raise NotImplementedError('Activation function [%s] is not implemented' % activation_name)

        self.fc_block = nn.Sequential(*self.fc_block)

    def forward(self, x):
        """
        Forward pass through the fully connected layer.

        Args:
        - x (tensor): Input tensor.

        Returns:
        - tensor: Output tensor after passing through the layer.
        """
        y = self.fc_block(x)
        return y


