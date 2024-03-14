# -*- coding: utf-8 -*-
"""
Created on Wed 01 Sept 2021

@author: Hakim Benkirane

    CentraleSupelec
    MICS laboratory
    9 rue Juliot Curie, Gif-Sur-Yvette, 91190 France

Build the Standard Autoencoder module.
"""
import torch
import torch.nn as nn


class AutoEncoder(nn.Module):
    """
    A simple Autoencoder model composed of an encoder and a decoder.

    Args:
    - encoder (nn.Module): The encoder module.
    - decoder (nn.Module): The decoder module.
    - device (str): The device to perform computations (e.g., 'cpu' or 'cuda').

    Methods:
    - __init__(self, encoder, decoder, device): Initializes the AutoEncoder instance.
    - _relocate(self): Moves the encoder and decoder to the specified device.
    - forward(self, x): Performs a forward pass through the autoencoder.
    - decode(self, z): Decodes the latent representation 'z' to reconstruct input 'x'.
    - loss(self, x, beta): Computes the reconstruction loss between input 'x' and its reconstruction 'x_hat'.
    """

    def __init__(self, encoder, decoder, device):
        """
        Initializes the AutoEncoder instance.

        Args:
        - encoder (nn.Module): The encoder module.
        - decoder (nn.Module): The decoder module.
        - device (str): The device to perform computations (e.g., 'cpu' or 'cuda').
        """
        super(AutoEncoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        self._relocate()

    def _relocate(self):
        """
        Moves the encoder and decoder to the specified device.
        """
        self.encoder.to(self.device)
        self.decoder.to(self.device)
        
    def forward(self, x):
        """
        Performs a forward pass through the autoencoder.

        Args:
        - x (Tensor): Input data.

        Returns:
        - x_hat (Tensor): Reconstructed data.
        - z (Tensor): Latent representation.
        """
        z = self.encoder(x)  
        x_hat = self.decoder(z)
        return x_hat, z

    def decode(self, z):
        """
        Decodes the latent representation 'z' to reconstruct input 'x'.

        Args:
        - z (Tensor): Latent representation.

        Returns:
        - x_hat (Tensor): Reconstructed data.
        """
        x_hat = self.decoder(z)
        return x_hat
        
    def loss(self, x, beta):
        """
        Computes the reconstruction loss between input 'x' and its reconstruction 'x_hat'.

        Args:
        - x (Tensor): Input data.
        - beta (float): Coefficient for the reconstruction loss.

        Returns:
        - reconstruction_loss (Tensor): Reconstruction loss value.
        """
        x_hat, z = self.forward(x)
        reconstruction_loss = nn.MSELoss()
        return reconstruction_loss(x, x_hat)





