import torch
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import pandas as pd
import torch.nn as nn
from torch.nn import functional as F
import sys
import shap
from sklearn import metrics
from scipy.stats import skew


def processPhenotypeDataForSamples(clinical_df, sample_id, le):
    """
    Process phenotype data for specific samples.

    Args:
    - clinical_df (DataFrame): Clinical data in the form of a DataFrame.
    - sample_id (list): List of sample identifiers.
    - le: LabelEncoder instance.

    Returns:
    - DataFrame: Processed phenotype data for the given samples.
    """
    phenotype = clinical_df.loc[sample_id, :]
    return phenotype


def randomTrainingSample(expr, sampleSize):
    """
    Create a random sample from expression data.

    Args:
    - expr (DataFrame): Expression data in the form of a DataFrame.
    - sampleSize (int): Size of the sample to be generated.

    Returns:
    - DataFrame: Randomly sampled expression data.
    """
    return expr.sample(n=sampleSize, axis=0, replace=True)


def splitExprandSample(condition, sampleSize, expr):
    """
    Split expression data based on a condition and create a random sample.

    Args:
    - condition: Condition for splitting the data.
    - sampleSize (int): Size of the sample to be generated.
    - expr (DataFrame): Expression data in the form of a DataFrame.

    Returns:
    - DataFrame: Split and randomly sampled expression data.
    """
    split_expr = expr[condition].sample(n=sampleSize, axis=0, replace=True)
    return split_expr


def printConditionalSelection(conditional, label_array):
    """
    Print counts based on a conditional selection.

    Args:
    - conditional: Condition for selection.
    - label_array: Array containing labels.
    """
    malecounts = label_array[conditional]
    unique, counts = np.unique(malecounts.iloc[:, 0], return_counts=True)


def addToTensor(expr_selection, device):
    """
    Convert selected data to a tensor.

    Args:
    - expr_selection (DataFrame): Selected data in the form of a DataFrame.
    - device: Torch device for tensor conversion.

    Returns:
    - Tensor: Converted data in tensor format.
    """
    selection = expr_selection.values.astype(dtype='float32')
    selection = torch.Tensor(selection).to(device)
    return selection


class ModelWrapper(nn.Module):
    """
    Wrapper class for the VAE model.

    Args:
    - vae_model: Variational Autoencoder model.
    - source (str): Source identifier.

    Methods:
    - forward(self, input): Forward pass method.
    """

    def __init__(self, vae_model, source):
        super(ModelWrapper, self).__init__()
        self.vae_model = vae_model
        self.source = source

    def forward(self, input):
        """
        Forward pass method.

        Args:
        - input: Input data.

        Returns:
        - Tensor: Result of the VAE model prediction.
        """
        return self.vae_model.source_predict(input, self.source)

