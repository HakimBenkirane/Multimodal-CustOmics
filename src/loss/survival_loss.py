import numpy as np
import torch
import torch.nn as nn


def CoxLoss(survtime, censor, hazard_pred, device, loss='cox-nnet', model=None, l2_reg=1e-2):
    """
    Computes the Cox loss function based on survival data.

    Args:
    - survtime (tensor): Tensor containing survival times.
    - censor (tensor): Tensor containing censoring information.
    - hazard_pred (tensor): Predicted hazard ratios.
    - device (torch.device): Device to perform computations on.
    - loss (str): Loss function type ('deepsurv' or 'cox-nnet').
    - model (nn.Module): Model to compute regularization if using 'deepsurv' loss.
    - l2_reg (float): L2 regularization strength.

    Returns:
    - tensor: Computed Cox loss based on the specified function.
    """
    if loss == 'deepsurv':
        nll_loss = NegativeLogLikelihood(l2_reg)
        return nll_loss(hazard_pred, survtime, censor, model)
    elif loss == 'cox-nnet':
        current_batch_len = len(survtime)
        R_mat = np.zeros([current_batch_len, current_batch_len], dtype=int)
        for i in range(current_batch_len):
            for j in range(current_batch_len):
                R_mat[i,j] = survtime[j] >= survtime[i]
        R_mat = torch.FloatTensor(R_mat).to(device)
        theta = hazard_pred.reshape(-1)
        exp_theta = torch.exp(theta)
        loss_cox = -torch.mean((theta - torch.log(torch.sum(exp_theta*R_mat, dim=1))) * censor)
        return loss_cox

class Regularization(object):
    """
    Class to apply regularization to model parameters.

    Args:
    - order (int): Order of the regularization.
    - weight_decay (float): Weight decay strength.

    Methods:
    - __call__(model): Applies regularization to the model's parameters.
    """
    def __init__(self, order, weight_decay):
        super(Regularization, self).__init__()
        self.order = order
        self.weight_decay = weight_decay

    def __call__(self, model):
        reg_loss = 0
        for name, w in model.named_parameters():
            if 'weight' in name:
                reg_loss = reg_loss + torch.norm(w, p=self.order)
        reg_loss = self.weight_decay * reg_loss
        return reg_loss

class NegativeLogLikelihood(nn.Module):
    """
    Negative Log-Likelihood loss function for survival analysis.

    Args:
    - l2_reg (float): L2 regularization strength.

    Methods:
    - forward(risk_pred, y, e, model): Computes the negative log-likelihood loss.
    """

    def __init__(self, l2_reg):
        super(NegativeLogLikelihood, self).__init__()
        self.L2_reg = l2_reg
        self.reg = Regularization(order=2, weight_decay=self.L2_reg)

    def forward(self, risk_pred, y, e, model):
        """
        Computes the negative log-likelihood loss for survival analysis.

        Args:
        - risk_pred (tensor): Predicted risk scores.
        - y (tensor): Tensor with observed time-to-event data.
        - e (tensor): Tensor with event indicators.
        - model (nn.Module): Model to compute regularization.

        Returns:
        - tensor: Computed negative log-likelihood loss with regularization.
        """
        mask = torch.ones(y.shape[0], y.shape[0])
        mask[(y.T - y) > 0] = 0
        log_loss = torch.exp(risk_pred) * mask
        log_loss = torch.sum(log_loss, dim=0) / torch.sum(mask, dim=0)
        log_loss = torch.log(log_loss).reshape(-1, 1)
        neg_log_loss = -torch.sum((risk_pred-log_loss) * e) / torch.sum(e)
        l2_loss = self.reg(model)
        return neg_log_loss + l2_loss