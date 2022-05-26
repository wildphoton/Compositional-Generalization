#!/usr/bin/env python
"""
Created by zhenlinx on 02/24/2022
"""
from .vae import VAE
from .rec_el import RecurrentEmergentLanguage
from .beta_tcvae import BetaTCVAE
from .ae import AutoEncoder

vae_models = {
    'VAE': VAE,
    'RecurrentEL': RecurrentEmergentLanguage,
    'BetaTCVAE': BetaTCVAE,
    'AE': AutoEncoder,
    }