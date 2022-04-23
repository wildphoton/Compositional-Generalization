#!/usr/bin/env python
"""
Created by zhenlinx on 02/24/2022
"""
import torch
from torch import nn
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
            ],
        'lstm_latent':
            {
                'hidden_size': 256
            }
    },
    'burgess_wide':{
        'encoder_cnn':
            [
                ('conv', (64, 4, 2, 1)),
                ('relu',),

                ('conv', (64, 4, 2, 1)),
                ('relu',),

                ('conv', (128, 4, 2, 1)),
                ('relu',),

                ('conv', (128, 4, 2, 1)),
                ('relu',),

                ('flatten', [1]),
            ],
        'encoder_latent':
            [
                ('linear', [512]),
                ('relu',),

                ('linear', [512]),
                ('relu',)
            ],
        'lstm_latent':
            {
                'hidden_size': 512
            }
    },
    'burgess_wide4x': {
        'encoder_cnn':
            [
                ('conv', (128, 4, 2, 1)),
                ('relu',),

                ('conv', (128, 4, 2, 1)),
                ('relu',),

                ('conv', (256, 4, 2, 1)),
                ('relu',),

                ('conv', (256, 4, 2, 1)),
                ('relu',),

                ('flatten', [1]),
            ],
        'encoder_latent':
            [
                ('linear', [1024]),
                ('relu',),

                ('linear', [1024]),
                ('relu',)
            ],
        'lstm_latent':
            {
                'hidden_size': 1024
            }
    },

    'base': {
        'encoder_cnn':
            [
                ('conv', (64, 4, 2, 1)),
                ('relu',),

                ('conv', (64, 4, 2, 1)),
                ('relu',),

                ('conv', (128, 4, 2, 1)),
                ('relu',),

                ('conv', (128, 4, 2, 1)),
                ('relu',),

                ('flatten', [1]),
            ],
        'encoder_latent':
            [
                ('linear', [512]),
                ('relu',),

                ('linear', [1024]),
                ('relu',),

                ('linear', [1024]),
                ('relu',),

                ('linear', [512]),
                ('relu',)
            ],
        'lstm_latent':
            {
                'hidden_size': 512
            }
    },

    'large': {
        'encoder_cnn':
            [
                ('conv', (128, 4, 2, 1)),
                ('relu',),

                ('conv', (128, 4, 2, 1)),
                ('relu',),

                ('conv', (256, 4, 2, 1)),
                ('relu',),

                ('conv', (256, 4, 2, 1)),
                ('relu',),

                ('flatten', [1]),
            ],
        'encoder_latent':
            [
                ('linear', [1024]),
                ('relu',),
                ('linear', [2048]),
                ('relu',),
                ('linear', [2048]),
                ('relu',),
                ('linear', [1024]),
                ('relu',)
            ],
        'lstm_latent':
            {
                'hidden_size': 1024
            }
    },
}

def build_architectures(input_size, name, latent_size, model, **kwargs):
    config = arch_configs[name]
    # build conv layers
    encoder_cnn_config = config['encoder_cnn'][:]
    encoder_cnn = FeedForward(input_size, encoder_cnn_config, flatten=False)
    if 'decoder_cnn' in config:
        decoder_cnn_config = config['decoder_config'][:]
    else:
        decoder_cnn_config = encoder_cnn_config[:]
        decoder_cnn_config = transpose_layer_defs(decoder_cnn_config, input_size)
    decoder_cnn = FeedForward(encoder_cnn.output_size, decoder_cnn_config, flatten=False)
    conv_layers = (encoder_cnn, decoder_cnn)

    # build latent layers
    if 'Recurrent' in model:
        # if it is recurrent models, the latent layers are built seperately with given config
        latent_layers = config['lstm_latent']
    else:
        encoder_latent_config = config['encoder_latent'][:]
        if model in ['VAE', 'BetaTCVAE']:
            # mu and log_variance
            encoder_latent_config += [('linear', [2 * latent_size])]
        elif model == 'DiscreteVAE' or model == 'AutoEncoder':
            # latent size = number of discrete code * n_classes
            encoder_latent_config += [('linear', [latent_size])]
        # elif model == 'DiscreteVAE':
        #     # latent size = number of discrete code * n_classes
        #     encoder_latent_config = encoder_latent_config[0]
        else:
            raise NotImplementedError('Not implemented for {}'.format(model))

        encoder_latent = FeedForward(encoder_cnn.output_size, encoder_latent_config, flatten=False)

        if 'decoder_latent' in config:
            decoder_latent_config = config['encoder_config'][:]
        else:
            decoder_latent_config = encoder_latent_config[:-1]
            if model in ['VAE', 'BetaTCVAE', 'AutoEncoder'] or model == 'DiscreteVAE':
                decoder_latent_config.append(('linear', [latent_size]))
            else:
                raise NotImplementedError(f'Not implemented for {model}')
            decoder_latent_config = transpose_layer_defs(decoder_latent_config, encoder_latent.input_size)
        decoder_latent = FeedForward(latent_size, decoder_latent_config, flatten=False)
        latent_layers = (encoder_latent, decoder_latent)

    return conv_layers, latent_layers

