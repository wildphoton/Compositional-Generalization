#!/usr/bin/env python
"""
Created by zhenlinx on 02/24/2022
"""
import torch
from architectures.cnn import build_cnn

def test_build_cnn():
    encoder_cnn, encoder_latent, decoder_latent, decoder_cnn = build_cnn((1, 64, 64), 'burgess', 4)
    x = torch.rand((2, 1, 64, 64))
    x = encoder_cnn(x)
    z = encoder_latent(x)
    y = decoder_latent(z[:, :4])
    y = decoder_cnn(y)
    pass

if __name__ == '__main__':
    test_build_cnn()
