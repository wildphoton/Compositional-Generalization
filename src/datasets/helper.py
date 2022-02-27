#!/usr/bin/env python
"""
Created by zhenlinx on 03/02/2021
"""
import os
import numpy as np
from torchvision import transforms

from .data_modules import DSpritesDataModule

def get_datamodule(name, data_dir, **configs):
    if 'dsprites' in name:
        return DSpritesDataModule(name, data_dir, **configs)
    else:
        raise ValueError()
    
if __name__ == '__main__':
    get_datamodule('dsprites90d_random', data_dir='../data')