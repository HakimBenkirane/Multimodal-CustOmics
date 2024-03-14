import torch.nn as nn
import torch.optim as optim
from src.metrics.survival import CIndex_lifeline
import torch
import numpy as np


class SurvivalNet(nn.Module):
    """
    Neural network model for survival analysis.

    Args:
    - config (dict): Dictionary containing configuration parameters for the network.

    Methods:
    - _build_network(): Method to construct the network architecture.
    - forward(X): Forward pass method.

    Attributes:
    - drop (float): Dropout rate.
    - norm (bool): Flag indicating batch normalization usage.
    - dims (list): List of dimensions for the network layers.
    - activation (str): Activation function used in the network.
    - device: Device type used for computations.
    - model: Sequential neural network model.
    """

    def __init__(self, config):
        super(SurvivalNet, self).__init__()
        # Parses parameters of the network from configuration
        self.drop = config['drop']
        self.norm = config['norm']
        self.dims = config['dims']
        self.activation = config['activation']
        self.device = config['device']
        # Builds the network
        self.model = self._build_network()

    def _build_network(self):
        """
        Method to construct the network architecture.

        Returns:
        - Sequential: Sequential neural network model.
        """
        layers = []
        for i in range(len(self.dims) - 1):
            if i and self.drop is not None:  # Adds dropout layer
                layers.append(nn.Dropout(self.drop))
            # Adds linear layer
            layers.append(nn.Linear(self.dims[i], self.dims[i + 1]))
            if self.norm:  # Adds batch normalize layer
                layers.append(nn.BatchNorm1d(self.dims[i + 1]))
            # Adds activation layer
            layers.append(eval('nn.{}()'.format(self.activation)))
        # Builds the sequential network
        return nn.Sequential(*layers)

    def forward(self, X):
        """
        Forward pass method.

        Args:
        - X: Input data.

        Returns:
        - Tensor: Output tensor after passing through the network.
        """
        return self.model(X)



