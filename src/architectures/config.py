#!/usr/bin/env python
"""
Created by zhenlinx on 02/24/2022
"""
def burgess():
    gm_type = 'lgm'
    latent_size = 10
    input_size = 1, 64, 64

    encoder_layers = [
        ('conv', (32, 4, 2, 1)),
        ('relu',),

        ('conv', (32, 4, 2, 1)),
        ('relu',),

        ('conv', (32, 4, 2, 1)),
        ('relu',),

        ('conv', (32, 4, 2, 1)),
        ('relu',),

        ('flatten', [1]),

        ('linear', [256]),
        ('relu',),

        ('linear', [256]),
        ('relu',)
    ]


def kim():
    gm_type = 'lgm'
    latent_size = 10
    input_size = 3, 64, 64

    encoder_layers = [
        ('conv', (32, 4, 2, 1)),
        ('relu',),

        ('conv', (32, 4, 2, 1)),
        ('relu',),

        ('conv', (64, 4, 2, 1)),
        ('relu',),

        ('conv', (64, 4, 2, 1)),
        ('relu',),

        ('flatten', [1]),

        ('linear', [256]),
        ('relu',),
    ]


def burgess_v2():
    gm_type = 'lgm'
    latent_size = 10
    input_size = 1, 64, 64

    encoder_layers = [
        ('conv', (32, 4, 2, 1)),
        ('relu',),

        ('conv', (32, 4, 2, 1)),
        ('relu',),

        ('conv', (64, 4, 2, 1)),
        ('relu',),

        ('conv', (64, 4, 2, 1)),
        ('relu',),

        ('flatten', [1]),

        ('linear', [256]),
        ('relu',),

        ('linear', [256]),
        ('relu',)
    ]


def mathieu():
    gm_type = 'lgm'
    latent_size = 10
    input_size = 1, 64, 64

    encoder_layers = [
        ('conv', (32, 4, 2, 1)),
        ('relu',),

        ('conv', (32, 4, 2, 1)),
        ('relu',),

        ('conv', (64, 4, 2, 1)),
        ('relu',),

        ('conv', (64, 4, 2, 1)),
        ('relu',),

        ('flatten', [1]),

        ('linear', [128]),
        ('relu',),

        # ('linear', [512]),
        # ('relu',),
    ]

    decoder_layers = [
        # ('linear', [512]),
        # ('relu',),

        ('linear', [128]),
        ('relu',),

        ('linear', [4 * 4 * 64]),
        ('relu',),

        ('unflatten', (64, 4, 4)),

        # ('tconv', (64, 4, 2, 1)),
        # ('relu',),

        ('tconv', (64, 4, 2, 1)),
        ('relu',),

        ('tconv', (32, 4, 2, 1)),
        ('relu',),

        ('tconv', (32, 4, 2, 1)),
        ('relu',),

        ('tconv', (1, 4, 2, 1)),
    ]


# Similar as above but with max-pooling
def mpcnn():
    gm_type = 'lgm'
    latent_size = 10
    input_size = 1, 64, 64

    encoder_layers = [
        # n-channels, size, stride, padding
        ('conv', (32, 3, 1, 1)),
        # size, stride, padding, type
        ('pool', (2, 2, 0, 'max')),
        ('relu',),

        ('conv', (32, 3, 1, 1)),
        ('pool', (2, 2, 0, 'max')),
        ('relu',),

        ('conv', (32, 3, 1, 1)),
        ('pool', (2, 2, 0, 'max')),
        ('relu',),

        ('flatten', [1]),

        ('linear', [256]),
        ('relu',),

        ('linear', [256]),
        ('relu',),
    ]

    decoder_layers = [
        ('linear', [256]),
        ('relu',),

        ('linear', [256]),
        ('relu',),

        ('linear', [2048]),
        ('relu',),

        ('unflatten', (32, 8, 8)),

        ('upsample', (16, 16)),
        ('tconv', (32, 3, 1, 1)),
        ('relu',),

        ('upsample', (32, 32)),
        ('tconv', (32, 3, 1, 1)),
        ('relu',),

        ('upsample', (64, 64)),
        ('tconv', (32, 3, 1, 1)),
    ]