#!/usr/bin/env python

import torch
from models.vae import VAE
from architectures.helper import build_architectures
from architectures.lstm_latent import LatentModuleLSTM
from commons.types_ import *


class RecurrentEmergentLanguage(VAE):
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
                 fix_length=False,
                 deterministic=False,
                 **kwargs):
        self.fix_length = fix_length
        self.deterministic = deterministic
        self.dictionary_size = dictionary_size
        self.gsm_temperature = gsm_temperature
        self.soft_discrete = soft_discrete

        super(RecurrentEmergentLanguage, self).__init__(
            input_size=input_size, architecture=architecture, latent_size=latent_size, beta=beta, recon_loss=recon_loss,
            lr=lr, optim=optim, weight_decay=weight_decay, **kwargs)

        # prior distributions (logP)
        prior = torch.log(
            torch.tensor([1 / self.dictionary_size] * self.dictionary_size, dtype=torch.float).repeat(
                1, self.latent_size, 1))
        self.register_buffer('prior', prior)


    def setup_models(self):
        (self.encoder_conv, self.decoder_conv), latent_config = build_architectures(
            self.input_size, self.architecture, self.latent_size, model=self.__class__.__name__)
        self.latent_layers = LatentModuleLSTM(
            input_size=self.encoder_conv.output_size,
            output_size=self.decoder_conv.input_size,
            hidden_size=latent_config['hidden_size'],
            latent_size=self.latent_size,
            dictionary_size=self.dictionary_size,
            fix_length=self.fix_length,
            temperature=self.gsm_temperature,
        )

    def encode(self, x: Tensor, sampling) -> Dict[str, Tensor]:
        feat = self.encoder_conv(x)
        res = self.latent_layers.encode(feat, sampling=sampling and not self.deterministic)
        return res

    def decode(self, inputs):
        return self.decoder_conv(self.latent_layers.decode(**inputs))

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
                return enc_res['z'].argmax(-1).float()
            elif mode == 'post':
                return self.latent_layers.decode(**enc_res)
            else:
                raise ValueError()

    def compute_KLD_loss(self, results):
        # Calculate KL divergence
        logits = results['logits']
        logits_dist = torch.distributions.OneHotCategorical(logits=logits)
        prior_batch = self.prior.expand(logits.shape)
        prior_dist = torch.distributions.OneHotCategorical(logits=prior_batch)
        kl = torch.distributions.kl_divergence(logits_dist, prior_dist)
        if not self.fix_length:
            eos_ind = results['eos_id']
            kl_loss_mask = torch.arange(0, self.latent_size).to(logits.device).repeat(
                logits.shape[0], 1) < eos_ind.unsqueeze(1)
            kl *= kl_loss_mask
        return kl.sum(1).mean(0)

    def make_backbone_name(self) -> str:
        """
        Get the name of the backbone according its parameters
        """
        return "{}_z{}{}{}_D{}_gsmT{}".format(
            self.architecture,
            self.latent_size,
            '_fix' if self.fix_length else '',
            '_determ' if self.deterministic else '',
            self.dictionary_size,
            self.gsm_temperature
        )

    def get_rep_size(self, mode):
        if mode == 'latent':
            return self.latent_size
        elif mode == 'pre':
            return self.encoder_conv.output_size
        elif mode == 'post':
            return self.latent_layers.output_size
        else:
            raise ValueError()