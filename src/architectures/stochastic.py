#!/usr/bin/env python
"""
Created by zhenlinx on 03/03/2022
"""
from commons.types_ import *
import torch
from torch.nn import functional as F

def gumble_softmax(logits: Tensor, temperature: float) -> Tensor:
    """
    Reparameterization trick to sample from a discrete distribution
    :param logits: (Tensor) [B x C X D] C is channels for features and D is for different discrete categories
    :param temperature: (Float) temperature for Gumble softmax
    :return: (Tensor) [B x C X D]
    """
    # sample from standard gumbel distribution
    g = torch.distributions.gumbel.Gumbel(torch.zeros_like(logits), torch.ones_like(logits))
    G = g.sample()
    return F.softmax((logits + G) / temperature, -1)


def straight_through_discretize(z_sampled_soft):
    """
    get argmax z (one-hot) from a sampled distribution with straight through gradient estimation
    :param p_sampled: distribution or logits of dicrete variables [B x C_d x D]
    :return: z_sampled_onehot: [B x C_d X D], z_sampled [B x C_d]
    """

    z_argmax = torch.argmax(z_sampled_soft, dim=-1, keepdim=True)
    z_argmax_one_hot = torch.zeros_like(z_sampled_soft).scatter_(-1, z_argmax, 1)

    # straight through gradient estimator
    z_sampled_onehot_with_grad = z_sampled_soft + (
            z_argmax_one_hot - z_sampled_soft).detach()

    return z_sampled_onehot_with_grad, z_argmax.squeeze(-1)