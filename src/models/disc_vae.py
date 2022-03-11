#!/usr/bin/env python
"""
Created by zhenlinx on 02/28/2022
"""
import os
import sys
sys.path.append(os.path.realpath('..'))
import torch

from models.types_ import *

from architectures.helper import build_architectures
from architectures.stochastic import gumble_softmax, straight_through_discretize
from models.vae import VAE

class DiscreteVAE(VAE):
    def __init__(self,
                 input_size: List,
                 architecture: str,
                 latent_size: int,
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
            input_size=input_size, architecture=architecture, latent_size=latent_size, beta=beta, recon_loss=recon_loss,
            lr=lr, optim=optim, weight_decay=weight_decay, **kwargs)

        # prior distributions (logP)
        prior = torch.log(
            torch.tensor([1 / self.dictionary_size] * self.dictionary_size, dtype=torch.float).repeat(
                1, self.latent_size, 1))
        self.register_buffer('prior', prior)


    def setup_models(self):
        (self.encoder_conv, self.decoder_conv), (self.encoder_latent, self.decoder_latent) = build_architectures(
            self.input_size, self.architecture, self.latent_size * self.dictionary_size, model=self.__class__.__name__)

    def reparameterize(self, latent: dict, sampling: bool) -> dict:
        logits = latent.reshape(-1, self.latent_size, self.dictionary_size)
        if sampling:
            # sample discrete code with gumble softmax
            z_soft = gumble_softmax(logits, self.gsm_temperature) # the sampled soft code (categorical distirbutions)
        else:
            z_soft = torch.softmax(logits, dim=-1)
        z, z_symbol = straight_through_discretize(z_soft) # discretize

        return {
            'logits': logits,
            'z_soft': z_soft,
            'z': z.flatten(1),
            'z_symbol': z_symbol,
        }

    def embed(self, x, mode, sampling=False, **kwargs):
        """
        Function to call to use VAE as a backbone model for downstream tasks
        :param x:
        :param mode:
        :param sampling:
        :param kwargs:
        :return:
        """
        if mode == 'pre':
            return self.encoder_conv(x)
        else:
            enc_res = self.encode(x, sampling=sampling)
            if mode == 'latent':
                return enc_res['z'].reshape(-1, self.latent_size, self.dictionary_size).argmax(-1).float()
            elif mode == 'post':
                return self.decoder_latent(enc_res['z'])
            else:
                raise ValueError()

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
            self.latent_size,
            self.dictionary_size,
        )

    def get_rep_size(self, mode):
        if mode == 'latent':
            return self.latent_size
        elif mode == 'pre':
            return self.encoder_conv.output_size
        elif mode == 'post':
            return self.decoder_latent.output_size
        else:
            raise ValueError()