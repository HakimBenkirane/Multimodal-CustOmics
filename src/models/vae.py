# -*- coding: utf-8 -*-
"""
Created on Wed 01 Sept 2021

@author: Hakim Benkirane

    CentraleSupelec
    MICS laboratory
    9 rue Juliot Curie, Gif-Sur-Yvette, 91190 France

Build the Variational Autoencoder module.
"""

import torch
import torch.nn as nn

from src.loss.mmd_loss import compute_mmd


class VAE(nn.Module):
    """
    Variational Autoencoder (VAE) model consisting of an encoder and a decoder.

    Args:
    - encoder (nn.Module): The encoder module.
    - decoder (nn.Module): The decoder module.
    - device (str): The device to perform computations (e.g., 'cpu' or 'cuda').

    Methods:
    - __init__(self, encoder, decoder, device): Initializes the VAE instance.
    - _relocate(self): Moves the encoder and decoder to the specified device.
    - reparameterization(self, mean, var): Applies reparameterization trick to sample from the latent space.
    - forward(self, x): Performs a forward pass through the VAE.
    - loss(self, x, beta): Computes the VAE loss, combining reconstruction loss and MMD (Maximum Mean Discrepancy).
    """

    def __init__(self, encoder, decoder, device):
        """
        Initializes the VAE instance.

        Args:
        - encoder (nn.Module): The encoder module.
        - decoder (nn.Module): The decoder module.
        - device (str): The device to perform computations (e.g., 'cpu' or 'cuda').
        """
        super(VAE, self).__init__()
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
        
    def reparameterization(self, mean, var):
        """
        Applies reparameterization trick to sample from the latent space.

        Args:
        - mean (Tensor): Mean of the latent space.
        - var (Tensor): Variance of the latent space.

        Returns:
        - z (Tensor): Sampled latent representation.
        """
        epsilon = torch.randn_like(var).to(self.device)            
        z = mean + var * epsilon                          
        return z
        
    def forward(self, x):
        """
        Performs a forward pass through the VAE.

        Args:
        - x (Tensor): Input data.

        Returns:
        - x_hat (Tensor): Reconstructed data.
        - z (Tensor): Latent representation.
        """
        mean, log_var = self.encoder(x)
        z = self.reparameterization(mean, torch.exp(0.5 * log_var))
        x_hat = self.decoder(z)
        return x_hat, z

    def loss(self, x, beta):
        """
        Computes the VAE loss, combining reconstruction loss and Maximum Mean Discrepancy (MMD).

        Args:
        - x (Tensor): Input data.
        - beta (float): Coefficient for the MMD term.

        Returns:
        - total_loss (Tensor): Combined VAE loss.
        """
        x_hat, z = self.forward(x)

        reconstruction_loss = nn.MSELoss()
        recon = reconstruction_loss(x, x_hat)

        true_samples = torch.randn(z.shape[0], z.shape[1]).to(self.device)
        MMD = torch.sum(compute_mmd(true_samples, z))

        return recon + beta * MMD



