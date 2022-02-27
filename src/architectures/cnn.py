#!/usr/bin/env python
"""
Created by zhenlinx on 02/24/2022
"""
import torch
from torch import nn
import sys
sys.path.append('..')
from architectures.feedforward import FeedForward, transpose_layer_defs


arch_configs= {
    'burgess':{
        'encoder_cnn':
            [
                ('conv', (32, 4, 2, 1)),
                ('relu',),

                ('conv', (32, 4, 2, 1)),
                ('relu',),

                ('conv', (64, 4, 2, 1)),
                ('relu',),

                ('conv', (64, 4, 2, 1)),
                ('relu',),

                ('flatten', [1]),
            ],
        'encoder_latent':
            [
                ('linear', [256]),
                ('relu',),

                ('linear', [256]),
                ('relu',)
            ]
    }
}

def build_cnn(input_size, name, latent_size):
    config = arch_configs[name]
    encoder_cnn_config = config['encoder_cnn'][:]
    encoder_cnn = FeedForward(input_size, encoder_cnn_config, flatten=False)
    encoder_latent_config = config['encoder_latent'][:]
    encoder_latent_config += [('linear', [2 * latent_size])]
    encoder_latent = FeedForward(encoder_cnn.output_size, encoder_latent_config, flatten=False)

    if 'decoder_layers' in config:
        decoder_cnn_config = config['encoder_config'][:]
        decoder_latent_config = config['encoder_config'][:]
    else:
        decoder_cnn_config = encoder_cnn_config[:]
        decoder_latent_config = encoder_latent_config[:-1]
        decoder_latent_config.append(('linear', [latent_size]))
        decoder_latent_config = transpose_layer_defs(decoder_latent_config, encoder_latent.input_size)
        decoder_cnn_config = transpose_layer_defs(decoder_cnn_config, input_size)

    decoder_latent = FeedForward(latent_size, decoder_latent_config, flatten=False)
    decoder_cnn = FeedForward(decoder_latent.output_size, decoder_cnn_config, flatten=False)

    return encoder_cnn, encoder_latent, decoder_latent, decoder_cnn

