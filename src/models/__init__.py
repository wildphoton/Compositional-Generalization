#!/usr/bin/env python
"""
Created by zhenlinx on 02/24/2022
"""
from .vae import VAE
from .disc_vae import DiscreteVAE

vae_models = {
    'VAE': VAE,
    'DiscreteVAE': DiscreteVAE,
    }