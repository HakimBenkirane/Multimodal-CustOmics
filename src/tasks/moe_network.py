# -*- coding: utf-8 -*-
"""
Created on Wed 01 Sept 2021

@author: Hakim Benkirane

    CentraleSupelec
    MICS laboratory
    9 rue Juliot Curie, Gif-Sur-Yvette, 91190 France

Build the Multi-Layer Classifier netowrk.
"""
import torch
import torch.nn as nn
from collections import OrderedDict
from src.encoders.encoder import Encoder
from src.tools.net_utils import FullyConnectedLayer
from src.loss.classification_loss import classification_loss

from torch.optim import Adam



class MultiTaskMOE(nn.Module):
    def __init__(self, pathways, spatial_clusters, n_class=2, latent_dim=256, norm_layer=nn.BatchNorm1d, leaky_slope=0.2, dropout=0,
                 hidden_dim=[128, 64]):
        super(MultiTaskMOE, self).__init__()

        # Initializing layers for the classifier
        self.pathways = pathways
        self.spatial_clusters = spatial_clusters
        self.classif_net = nn.Sequential(nn.Linear(latent_dim, n_class), nn.Softmax(dim=1))
        self.survival_net = FullyConnectedLayer(latent_dim, 1, norm_layer=norm_layer,
                                                           leaky_slope=leaky_slope, dropout=dropout, activation=True, activation_name='Softmax')
        self.gating_network_classif = nn.Sequential(Encoder(input_dim=len(self.pathways) * len(self.spatial_clusters) * latent_dim, 
                                      hidden_dim=hidden_dim, latent_dim=len(self.pathways) * len(self.spatial_clusters)), nn.Softmax(dim=1))
        self.gating_network_surv = nn.Sequential(Encoder(input_dim=len(self.pathways) * len(self.spatial_clusters) * latent_dim, 
                                      hidden_dim=hidden_dim, latent_dim=len(self.pathways) * len(self.spatial_clusters)), nn.Softmax(dim=1))
        self.latent_dim = latent_dim
        self._register()
        
    def _register(self):
        for pathway in self.pathways:
            for cluster in self.spatial_clusters:
                for name, param in self.classif_net.named_parameters():
                    name = "classif_net_" + name.replace(".", " ")
                    self.register_parameter(name=name, param=param)
                for name, param in self.survival_net.named_parameters():
                    name = "survival_net_" + name.replace(".", " ")
                    self.register_parameter(name=name, param=param)
        
    def forward(self, multimodal_rep):
        classif_result = 0
        surv_result = 0
        reps_concat = []
        for pathway in self.pathways:
            for cluster in self.spatial_clusters:
                reps_concat.append(multimodal_rep[pathway][cluster])
        self.weights_classif = self.gating_network_classif(torch.cat(reps_concat, axis=1))
        self.weights_surv = self.gating_network_surv(torch.cat(reps_concat, axis=1))
        multimodal_classif_rep = 0
        multimodal_surv_rep = 0
        for i, pathway in enumerate(self.pathways):
            for j, cluster in enumerate(self.spatial_clusters):
                multimodal_classif_rep += self.weights_classif[:, i*j].unsqueeze(1) * multimodal_rep[pathway][cluster] 
                multimodal_surv_rep += self.weights_surv[:, i*j].unsqueeze(1) * multimodal_rep[pathway][cluster] 
        classif_result = self.classif_net(multimodal_classif_rep)
        surv_result = self.survival_net(multimodal_surv_rep)
        return classif_result, surv_result

