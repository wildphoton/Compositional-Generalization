#!/usr/bin/env python
"""
Created by zhenlinx on 02/24/2022
"""
import os
import sys
sys.path.append(os.path.realpath('..'))

import torch
from architectures.helper import build_architectures
from architectures.lstm_latent import LatentModuleLSTM

def test_build_cnn():
    (encoder_cnn, decoder_cnn), (encoder_latent, decoder_latent) = build_architectures((1, 64, 64), 'burgess_base', 10, 'VAE')
    x = torch.rand((2, 1, 64, 64))
    x = encoder_cnn(x)
    z = encoder_latent(x)
    y = decoder_latent(z[:, :4])
    y = decoder_cnn(y)


def test_lstm_latent():
    encoder_latent = LatentModuleLSTM(256, 256, 128, 8, 5)
    feat = torch.rand(2, 256)
    outputs = encoder_latent(feat)
    pass

if __name__ == '__main__':
    test_build_cnn()
    test_lstm_latent()