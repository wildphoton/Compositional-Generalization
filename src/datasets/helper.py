#!/usr/bin/env python
"""
Created by zhenlinx on 03/02/2021
"""
import os
import numpy as np
from torchvision import transforms

from .data_modules import DSpritesDataModule, MPI3DDataModule

def get_datamodule(name, data_dir, **configs):
    if 'dsprites' in name:
        return DSpritesDataModule(name=name, data_dir=data_dir, **configs)
    elif 'mpi3d' in name:
        return MPI3DDataModule(name=name, data_dir=data_dir, **configs)
    else:
        raise ValueError()
