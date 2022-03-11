#!/usr/bin/env python
"""
Created by zhenlinx on 01/12/2022
"""
import os
import sys
sys.path.append(os.path.realpath('..'))

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

def testBetaVAEICC():
    model = VAE(
        input_size=(1, 64, 64),
        architecture='burgess',
        latent_size=8,
        beta=1,
        icc=True,
        icc_max=25,
        icc_steps=100000,
    ).cuda()
    x = torch.rand((2, 1, 64, 64)).cuda()
    with torch.no_grad():
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
    with torch.no_grad():
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
    print(model.name)
    x = torch.rand((2, 1, 64, 64))
    output = model.step((x, None), 0)
    output = model.embed(x, 'pre')
    output = model.embed(x, 'post')
    output = model.embed(x, 'latent')
    pass

if __name__ == '__main__':
    # testVAE()
    # testBetaVAEICC()
    # testDiscVAE()
    testRecDiscVAE()