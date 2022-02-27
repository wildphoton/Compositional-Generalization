#!/usr/bin/env python
"""
Created by zhenlinx on 05/05/2021
refer to the code at https://github.com/rmccorm4/PyTorch-LMDB
"""
import os
import pickle
import lmdb
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import pytorch_lightning as pl



class MultiDsprites(Dataset):
    """
    The Multi-Dsprites dataset (colored sprites and background version).
    The original dataset was release at https://github.com/deepmind/multi_object_datasets#multi-dsprites

    Following are some statistics of image features

    # of objects:
    [1., 2., 3., 4.] : [250009, 249009, 250747, 250235]

    object shapes
    [0., 1., 2., 3.] : [1000000, 1333367, 1333130, 1333503]

    object colors: 763

    orientations: 77
    [0.        , 0.04027677, 0.08055353, 0.12083042, 0.16110732,
       0.20138407, 0.24166083, 0.28193748, 0.32221463, 0.36249137,
       0.40276814, 0.44304502, 0.48332193, 0.5235988 , 0.56387544,
       0.6041521 , 0.64442927, 0.6847061 , 0.72498274, 0.7652596 ,
       0.80553657, 0.8458134 , 0.88609004, 0.92636716, 0.96664387,
       1.0069207 , 1.0471976 , 1.0874742 , 1.1277512 , 1.168028  ,
       1.2083046 , 1.2485818 , 1.2888585 , 1.3291353 , 1.3694122 ,
       1.4096888 , 1.4499658 , 1.4902426 , 1.5305192 , 1.6110731 ,
       1.6916268 , 1.7721804 , 1.8527339 , 1.9332877 , 2.0138414 ,
       2.0943952 , 2.1749485 , 2.2555025 , 2.336056  , 2.4166098 ,
       2.4971635 , 2.577717  , 2.6582706 , 2.7388244 , 2.8193781 ,
       2.8999317 , 2.9804852 , 3.061039  , 3.2221463 , 3.3832536 ,
       3.5443609 , 3.7054682 , 3.8665755 , 4.027683  , 4.1887903 ,
       4.3498974 , 4.511005  , 4.672112  , 4.8332195 , 4.9943266 ,
       5.155434  , 5.316541  , 5.4776487 , 5.6387563 , 5.7998633 ,
       5.960971  , 6.122078  ],

    scales:
    [0.5, 0.6, 0.7, 0.8, 0.9, 1. ]

    x/y: 32
    [0.        , 0.03225806, 0.06451613, 0.09677419, 0.12903225,
       0.16129032, 0.19354838, 0.22580644, 0.2580645 , 0.29032257,
       0.32258064, 0.3548387 , 0.38709676, 0.41935483, 0.4516129 ,
       0.48387095, 0.516129  , 0.5483871 , 0.58064514, 0.61290324,
       0.6451613 , 0.67741936, 0.7096774 , 0.7419355 , 0.7741935 ,
       0.8064516 , 0.83870965, 0.87096775, 0.9032258 , 0.9354839 ,
       0.9677419 , 1.        ]

    meta_class include three groups of class labels:
    'n_objects': number of objects 4-classes: [1,2,3,4]
    'shape': 3 multi-label classification-if a image contains a shape-X object
    'scale': 6 multi-label classification-if a image contains a scale-X object

    """
    ATTRIBUTES = ['x', 'y', 'shape', 'color', 'visibility', 'orientation', 'scale']
    CLASS_ATTRIBUTES = ['shape', 'scale', 'n_objects']
    NUM_SAMPLES = int(1E6)
    NUM_CLASSES = np.array([4, 2, 2, 2, 2, 2, 2, 2, 2, 2])
    def __init__(self, root, range=None, use_latent_class=True, transform=None, n_samples=None, **kwargs):
        super(MultiDsprites, self).__init__()
        self.image_file = os.path.join(root, "multi_dsprites_colored_on_colored_image.npy")
        # self.metadata_file = os.path.join(root, "multi_dsprites_colored_on_colored_metadata.npz")
        self.metaclass_file = os.path.join(root, "multi_dsprites_colored_on_colored_metaclass.npz")

        print('loading data')
        meta_class_zip = np.load(self.metaclass_file, allow_pickle=True)
        self.images = np.load(self.image_file).transpose([0, 3, 1, 2])
        print('loading data done')

        self.meta_labels = torch.from_numpy(
            np.concatenate([meta_class_zip[key] for key in meta_class_zip], axis=1)).long()
        self.range = range
        if self.range is not None:
            self.raw_num_samples = len(self.range)
        else:
            self.raw_num_samples = self.NUM_SAMPLES

        self.transform = transform
        self.n_samples = n_samples if n_samples is not None else self.raw_num_samples

    def __getitem__(self, idx):
        """Get the image of `idx`
        Return
        ------
        sample : torch.Tensor
            Tensor in [0.,1.] of shape `img_size`.

        labels: Not Implemented

        """
        # map the recursive id to real
        idx = idx % self.raw_num_samples
        if self.range is not None:
            idx = self.range[idx]

        img = self.images[idx]

        if self.transform is not None:
            img = self.transform(img)

        if self.transform is not None:
            img = self.transform(img)
        img = torch.from_numpy(img).float()
        img = (img / 255.0 - 0.5) / 0.5

        return img, self.meta_labels[idx]

    def __len__(self):
        return self.n_samples



if __name__ == '__main__':
    from tqdm import tqdm
    root = "/playpen-raid2/zhenlinx/Data/multi-objects/multi-dsprites"
    data = MultiDsprites(root=root)
    print("Benchmarking MultiDspritesLMDB dataset")
    for i in tqdm(np.random.choice(1000000, 100000)):
        sample = data[i]

    print("Benchmarking MultiDspritesLMDB dataset with 8 worker loader")
    loader = DataLoader(data,
                      batch_size=512,
                      shuffle=True,
                      drop_last=False,
                      num_workers=10,
                      pin_memory=False
                      )
    i = 0
    for data in tqdm(loader):
        i+=1
        if i > 10000:
            break
        pass
    pass