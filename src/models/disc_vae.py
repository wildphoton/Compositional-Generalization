#!/usr/bin/env python
"""
Created by zhenlinx on 02/28/2022
"""
import os
import sys
sys.path.append(os.path.realpath('..'))
from math import sqrt
import torch
import torch.nn as nn
from torch.nn import functional as F
import torchvision.utils as vutils

from .types_ import *
import pytorch_lightning as pl

from architectures import build_cnn
from evaluation.dci import DCIMetrics
from .optimizer import init_optimizer, init_lr_scheduler
from .vae import VAE

class DiscreteVAE(VAE):
    def __init__(self,
                 input_size: List,
                 architecture: str,
                 latent_dim: int,
                 dictionary_size: int,
                 beta: float = 1.0,
                 recon_loss: str = 'mse',
                 gsm_temperature: int = 1,
                 soft_discrete=False,
                 lr: float = 0.001,
                 optim: str = 'adam',
                 weight_decay: float = 0,
                 **kwargs) -> None:

        self.dictionary_size = dictionary_size
        self.gsm_temperature = gsm_temperature
        self.soft_discrete = soft_discrete

        super(DiscreteVAE, self).__init__(
            input_size, architecture, latent_dim, beta, recon_loss, lr, optim, weight_decay, **kwargs)

        # prior distributions (logP)
        prior = torch.log(
            torch.tensor([1 / self.dictionary_size] * self.dictionary_size, dtype=torch.float).repeat(
                1, self.latent_dim, 1))
        self.register_buffer('prior', prior)


    def setup_models(self):
        self.encoder_conv, self.encoder_latent, self.decoder_latent, self.decoder_conv = build_cnn(
            self.input_size, self.architecture, self.latent_dim*self.dictionary_size, model=self.__class__.__name__)

    def reparameterize(self, latent: dict, sampling: bool) -> dict:
        logits = latent.reshape(-1, self.latent_dim, self.dictionary_size)
        if sampling:
            # sample discrete code with gumble softmax
            z_soft = self.gumble_softmax(logits) # the sampled soft code (categorical distirbutions)
        else:
            z_soft = torch.softmax(logits, dim=-1)
        z, z_symbol = self.straight_through_discretize(z_soft) # discretize

        return {
            'logits': logits,
            'z_soft': z_soft,
            'z': z.flatten(1),
            'z_symbol': z_symbol,
        }

    def gumble_softmax(self, logits: Tensor) -> Tensor:
        """
        Reparameterization trick to sample from a discrete distribution
        :param logits: (Tensor) [B x C X D] C is channels for features and D is for different discrete categories
        :return: (Tensor) [B x C X D]
        """
        # sample from standard gumbel distribution
        g = torch.distributions.gumbel.Gumbel(torch.zeros_like(logits), torch.ones_like(logits))
        G = g.sample()
        return F.softmax((logits + G) / self.gsm_temperature, -1)

    def straight_through_discretize(self, z_sampled_soft):
        """
        get argmax z (one-hot) from a sampled distribution with straight through gradient estimation
        :param p_sampled: distribution or logits of dicrete variables [B x C_d x D]
        :return: z_sampled_onehot: [B x C_d X D], z_sampled [B x C_d]
        """

        z_argmax = torch.argmax(z_sampled_soft, dim=-1, keepdim=True)

        z_argmax_one_hot = torch.zeros_like(z_sampled_soft).scatter_(-1, z_argmax, 1)
        z_sampled_onehot_with_grad = z_sampled_soft + (
                z_argmax_one_hot - z_sampled_soft).detach()  # straight through gradient estimator
        return z_sampled_onehot_with_grad, z_argmax.squeeze(-1)

    def compute_KLD_loss(self, results):
        # Calculate KL divergence
        logits = results['logits']
        logits_dist = torch.distributions.OneHotCategorical(logits=logits)
        prior_batch = self.prior.expand(logits.shape)
        prior_dist = torch.distributions.OneHotCategorical(logits=prior_batch)
        kl = torch.distributions.kl_divergence(logits_dist, prior_dist).sum(1).mean(0)
        return kl

    def make_backbone_name(self) -> str:
        """
        Get the name of the backbone according its parameters
        """
        return "{}_z{}_D{}".format(
            self.architecture,
            self.latent_dim,
            self.dictionary_size,
        )

    def get_rep_size(self, mode):
        if mode == 'latent':
            return self.latent_dim * self.dictionary_size
        elif mode == 'pre':
            return self.encoder_conv.output_size
        elif mode == 'post':
            return self.decoder_latent.output_size
        else:
            raise ValueError()