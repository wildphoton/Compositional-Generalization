#!/usr/bin/env python
"""
Created by zhenlinx on 01/12/2022
"""
import torch
from models.vae import VAE
from models.disc_vae import DiscreteVAE
from models.rec_disc_vae import RecurrentDiscreteVAE

def testVAE():
    model = VAE(
        input_size=(1, 64, 64),
        architecture='burgess',
        latent_size=8,
        beta=1,
    )
    x = torch.rand((2, 1, 64, 64))
    output = model.step((x, None), 0)
    output = model.embed(x, 'pre')
    output = model.embed(x, 'post')
    output = model.embed(x, 'latent')
    pass

def testDiscVAE():
    model = DiscreteVAE(
        input_size=(1, 64, 64),
        architecture='burgess',
        latent_size=8,
        beta=0,
        dictionary_size=128,
    )
    x = torch.rand((2, 1, 64, 64))
    output = model.step((x, None), 0)
    output = model.embed(x, 'pre')
    output = model.embed(x, 'post')
    output = model.embed(x, 'latent')
    pass

def testRecDiscVAE():
    model = RecurrentDiscreteVAE(
        input_size=(1, 64, 64),
        architecture='burgess',
        latent_size=8,
        beta=0,
        dictionary_size=128,
    )
    x = torch.rand((2, 1, 64, 64))
    output = model.step((x, None), 0)
    output = model.embed(x, 'pre')
    output = model.embed(x, 'post')
    output = model.embed(x, 'latent')
    pass

if __name__ == '__main__':
    # testVAE()
    # testDiscVAE()
    testRecDiscVAE()