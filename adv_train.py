"""
Adversarial training
"""

import copy
import numpy as np
from collections import Iterable
from scipy.stats import truncnorm

import torch
import torch.nn as nn

from attack import FGSMAttack, LinfPGDAttack



def adv_train(X, model, adversary):
    """
    Adversarial training. Returns perturbed mini batch.
    """

    # If adversarial training, need a snapshot of
    # the model at each batch to compute grad, so
    # as not to mess up with the optimization step
    model_cp = copy.deepcopy(model)
    for p in model_cp.parameters():
        p.requires_grad = False
    model_cp.eval()

    adversary.model = model_cp

    X_adv = adversary.perturb(X.cpu().numpy())

    return torch.from_numpy(X_adv)


# def FGSM_train_rnd(X, y, model, criterion, fgsm_adversary, epsilon_max=0.3):
#     """
#     FGSM with epsilon sampled from a truncated normal distribution.
#     Returns pertubed mini batch.
#     Kurakin et al., ADVERSARIAL MACHINE LEARNING AT SCALE, 2016
#     """
#
#     # If adversarial training, need a snapshot of
#     # the model at each batch to compute grad, so
#     # as not to mess up with the optimization step
#     model_cp = copy.deepcopy(model)
#     for p in model_cp.parameters():
#         p.requires_grad = False
#     model_cp.eval()
#
#     fgsm_adversary.model = model_cp
#
#     # truncated Gaussian
#     m = X.size()[0]  # mini-batch size
#     mean, std = 0., epsilon_max / 2
#     epsilons = np.abs(truncated_normal(mean, std, m))[:, np.newaxis, np.newaxis, np.newaxis]
#
#     X_adv = fgsm_adversary.perturb(X.numpy(), y, epsilons)
#
#     return torch.from_numpy(X_adv)