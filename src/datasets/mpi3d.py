#!/usr/bin/env python
"""
Created by zhenlinx on 03/20/2022
"""
import os
import numpy as np
from urllib import request
from itertools import product

import torch
from torch.utils.data import Dataset
import torchvision.transforms as trans

class MPI3D(Dataset):
    """
    #==========================================================================
    # Latent Dimension,    Latent values                                 N vals
    #==========================================================================

    # object color:        white=0, green=1, red=2, blue=3,                  6
    #                      brown=4, olive=5
    # object shape:        cone=0, cube=1, cylinder=2,                       6
    #                      hexagonal=3, pyramid=4, sphere=5
    # object size:         small=0, large=1                                  2
    # camera height:       top=0, center=1, bottom=2                         3
    # background color:    purple=0, sea green=1, salmon=2                   3
    # horizontal axis:     40 values liearly spaced [0, 39]                 40
    # vertical axis:       40 values liearly spaced [0, 39]                 40
    """
    files = {"toy": "mpi3d_toy.npz",
             "realistic": "mpi3d_realistic.npz",
             "real": "mpi3d_real.npz"}
    urls = {
        "toy": 'https://storage.googleapis.com/disentanglement_dataset/Final_Dataset/mpi3d_toy.npz',
        "realistic": 'https://storage.googleapis.com/disentanglement_dataset/Final_Dataset/mpi3d_realistic.npz',
        "real": 'https://storage.googleapis.com/disentanglement_dataset/Final_Dataset/mpi3d_real.npz'
    }
    task_types = np.array(['cls', 'cls', 'cls', 'reg', 'cls', 'reg', 'reg',])
    num_factors = 7
    # 'object_color', 'object_shape', 'object_size', 'camera_height', 'background_color', 'horizontal_axis', 'vertical_axis'
    lat_names = ('color', 'shape', 'size', 'height', 'bg_color', 'x-axis', 'y-axis')
    lat_sizes = np.array([6, 6, 2, 3, 3, 40, 40])
    img_size = (3, 64, 64)
    total_sample_size = 1036800
    NUM_CLASSES = list(lat_sizes)

    lat_values = {'color': np.arange(6),
                  'shape': np.arange(6),
                  'size': np.arange(2),
                  'height': np.arange(3),
                  'bg_color': np.arange(3),
                  'x-axis': np.arange(40),
                  'y-axis': np.arange(40)}

    def __init__(self, root, subset, range=None, n_samples=None):
        self.root = root
        self.subset = subset
        self.file_path = os.path.join(root, self.files[subset])
        if not os.path.exists(self.file_path):
            self.download(self.file_path)
        self.imgs = np.load(self.file_path)['images']
        latent_values = np.asarray(list(product(*self.lat_values.values())), dtype=np.int8)
        self.latent_values = torch.from_numpy(latent_values)

        if range is not None:
            self.imgs = self.imgs[range]
            self.latent_values = self.latent_values[range]

        # self.imgs = torch.from_numpy(self.imgs)
        image_transforms = [
            trans.ToTensor(),
            trans.ConvertImageDtype(torch.float32),
        ]
        # if color_mode == 'hsv':
        #     image_transforms.insert(0, trans.Lambda(rgb2hsv))

        self.transform = trans.Compose(image_transforms)
        self.n_samples = n_samples if n_samples is not None else len(self.imgs)
        self.raw_num_samples = len(self.imgs)


    def __getitem__(self, idx):
        # map the recursive id to real id
        idx = idx % self.raw_num_samples

        img, label = self.imgs[idx], self.latent_values[idx]
        if self.transform:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return self.n_samples

    def download(self, file_path):
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        print('downloading MPI3D {}'.format(self.subset))
        request.urlretrieve(self.urls[self.subset], file_path)
        print('download complete')

    def sample(self, num, random_state):
        indices = random_state.choice(self.raw_num_samples,
                                      num,
                                      replace=False if self.raw_num_samples > num else True)
        factors = self.latent_values[indices].numpy().astype(np.int32)
        samples = self.imgs[indices]
        if np.issubdtype(samples.dtype, np.uint8):
            samples = samples.astype(np.float32) / 255.
        return factors, samples
