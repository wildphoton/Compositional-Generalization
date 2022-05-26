#!/usr/bin/env python
"""
Created by zhenlinx on 05/11/2021
"""
import os
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import pytorch_lightning as pl

from .dsprites import DSprites
from .multi_dsprites import MultiDsprites
from .mpi3d import MPI3D

DATASETS_DICT = {
                # "mnist": "MNIST",
#                  "fashion": "FashionMNIST",
                 "dsprites": DSprites,
                 "dsprites90d": DSprites,
                 "multidsprites": MultiDsprites,
                 "mpi3d": MPI3D,
                 # "celeba": "CelebA",
                 # "celeba": "CelebA",
                 # "chairs": "Chairs"
}


class MetaDataModule(pl.LightningDataModule):
    def __init__(self, name, data_dir, batch_size: int = 128, num_workers=4, n_train=None, n_fold=1,
                 random_seed=None, virtual_n_samples=None, **dataset_config):
        super(MetaDataModule, self).__init__()
        self.name = name
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.transform = None
        self.num_workers = num_workers
        self.n_train = n_train  # use only partial of training set
        self.n_fold = n_fold  # the dataset will be n_fold x n_train samples
        self.virtual_n_samples = virtual_n_samples  # change the number of samples to be total training steps for prefetching purpose
        self.dataset_config = dataset_config

        self.class_name = self.name.split('_')[0]
        self.dataset_class = DATASETS_DICT[self.class_name]
        self.num_classes = self.dataset_class.NUM_CLASSES # unique classes for each attribute

        self.random_seed = random_seed

        self.train_ind = None
        self.test_ind = None
        self.train_dataset = None
        self.test_dataset = None
        
    def prepare_data(self) -> None:
        raise NotImplementedError()

    def train_dataloader(self, shuffle=True):
        return DataLoader(self.train_dataset,
                          batch_size=self.batch_size,
                          shuffle=shuffle,
                          drop_last=False,
                          num_workers=self.num_workers,
                          pin_memory=True,
                          persistent_workers=self.num_workers>0,
                          )

    def val_dataloader(self):
        return self.test_dataloader()

    def test_dataloader(self):
        return DataLoader(self.test_dataset,
                          batch_size=self.batch_size,
                          shuffle=False,
                          drop_last=False,
                          num_workers=self.num_workers)

class DSpritesDataModule(MetaDataModule):
    def prepare_data(self):
        # the name is formated as CLASSNAME_MODE_VERSION
        self.mode = self.name.split('_')[1]
        self.version = self.name.split('_')[2]
        if self.class_name == 'dsprites':

            if self.mode == 'element':
                range_test = [[1, ], [0, 1], np.arange(13, 26), np.arange(21, 32), np.arange(21, 32)]
            elif self.mode == 'range':
                range_test = [[0, ], np.arange(6), np.arange(40), np.arange(16, 32), np.arange(32)]
            elif self.mode == 'all':
                range_test = None
            else:
                raise ValueError('Undefined splitting')

            range_all = [np.arange(3), np.arange(6), np.arange(40), np.arange(32), np.arange(32)]
            if range_test:
                self.train_ind, self.test_ind = DSprites.get_partition(range_all, range_test)
            else:
                self.train_ind = DSprites.get_partition(range_all, range_test)
                self.test_ind = self.train_ind

        elif self.class_name == 'dsprites90d':
            range_all = [np.arange(3), np.arange(6), np.arange(10), np.arange(32), np.arange(32)]
            if self.mode == 'element':
                if self.version == 'v1':
                    range_test = [[0, ], [0, 1], np.arange(6, 10), np.arange(21, 32), np.arange(21, 32)]
                elif self.version == 'v2':
                    range_test = [[1, ], [0, 1], np.arange(6, 10), np.arange(21, 32), np.arange(21, 32)]
                elif self.version == 'v3':
                    range_test = [[2, ], [0, 1], np.arange(6, 10), np.arange(21, 32), np.arange(21, 32)]
                else:
                    raise ValueError('Undefined splitting')

            elif self.mode == 'multiElement':
                if self.version == 'v1':  # train/test: 179712/4608
                    range_test = (
                        [[0, ], [0, 1], np.arange(1, 4), np.arange(16, 32), np.arange(16, 32)],
                        [[1, ], [2, 3], np.arange(4, 7), np.arange(0, 16), np.arange(0, 16)],
                        [[2, ], [4, 5], np.arange(7, 10), np.arange(16, 32), np.arange(0, 16)],
                    )
                elif self.version == 'v2':  # train/test: 173568/10752
                    range_test = (
                        [[0, ], [0, 1, 2], np.arange(0, 5), np.arange(16, 32), np.arange(16, 32)],
                        [[1, ], [3, 4, 5], np.arange(6, 10), np.arange(0, 16), np.arange(0, 16)],
                        [[2, ], [1, 2, 3], np.arange(3, 8), np.arange(8, 24), np.arange(8, 24)],
                    )
                elif self.version == 'v3':  # 92160/92160
                    range_test = (
                        [[0, ], np.arange(6), np.arange(10), np.arange(16, 32), np.arange(32)],
                        [[1, ], np.arange(6), np.arange(10), np.arange(0, 16), np.arange(32)],
                        [[2, ], np.arange(6), np.arange(10), np.arange(8, 24), np.arange(32)]
                    )
                elif self.version == 'v4':  # 138240/46080
                    range_test = (
                        [[0, ], np.arange(6), np.arange(10), np.arange(16, 32), np.arange(8, 24)],
                        [[1, ], np.arange(6), np.arange(10), np.arange(0, 16), np.arange(16, 32)],
                        [[2, ], np.arange(6), np.arange(10), np.arange(8, 24), np.arange(0, 16)]
                    )
                elif self.version == 'v5':  # 156672/27648
                    range_test = (
                        [[0, ], np.arange(6), np.arange(1, 4), np.arange(16, 32), np.arange(32)],
                        [[1, ], np.arange(6), np.arange(4, 7), np.arange(0, 16), np.arange(32)],
                        [[2, ], np.arange(6), np.arange(7, 10), np.arange(8, 24), np.arange(32)]
                    )
                elif self.version == 'v6':  # 165888/18432
                    range_test = (
                        [[0, ], np.arange(0, 2), np.arange(1, 4), np.arange(32), np.arange(32)],
                        [[1, ], np.arange(2, 4), np.arange(4, 7), np.arange(32), np.arange(32)],
                        [[2, ], np.arange(4, 6), np.arange(7, 10), np.arange(32), np.arange(32)]
                    )
                else:
                    raise ValueError('Undefined splitting')
            elif self.mode == 'range':
                # shape + posX
                if self.version == 'v1':
                    range_test = [[0, ], np.arange(6), np.arange(10), np.arange(16, 32), np.arange(32)]
                elif self.version == 'v2':
                    range_test = [[1, ], np.arange(6), np.arange(10), np.arange(16, 32), np.arange(32)]
                elif self.version == 'v3':
                    range_test = [[2, ], np.arange(6), np.arange(10), np.arange(16, 32), np.arange(32)]

                # scale + posX to check if shape has too low diversity
                elif self.version == 'v4':
                    range_test = [[0, 1, 2], np.arange(0, 2), np.arange(10), np.arange(16, 32), np.arange(32)]
                elif self.version == 'v5':
                    range_test = [[0, 1, 2], np.arange(2, 4), np.arange(10), np.arange(16, 32), np.arange(32)]
                elif self.version == 'v6':
                    range_test = [[0, 1, 2], np.arange(4, 6), np.arange(10), np.arange(16, 32), np.arange(32)]

                # orientation + posX to check if shape has too low diversity
                elif self.version == 'v7':
                    range_test = [[0, 1, 2], np.arange(6), np.arange(1, 4), np.arange(16, 32), np.arange(32)]
                elif self.version == 'v8':
                    range_test = [[0, 1, 2], np.arange(6), np.arange(4, 7), np.arange(16, 32), np.arange(32)]
                elif self.version == 'v9':
                    range_test = [[0, 1, 2], np.arange(6), np.arange(7, 10), np.arange(16, 32), np.arange(32)]
                else:
                    raise ValueError('Undefined version')
            elif self.mode == 'random':
                # 184320 total images
                test_sizes = {
                    'v1': 30000,  # 5: 1
                    'v2': 60000,  # 2: 1
                    'v3': 90000,  # 1: 1
                    'v4': 129024,  # 3: 7
                    'v5': 165888,  # 1: 9
                    'v6': 175104,   # 5: 95 n_train = 9216
                    # 'v7': 182476,  # 1: 99
                }
                test_size = test_sizes[self.version]
                # total 184K
            else:
                raise ValueError('Undefined splitting')

            if self.mode != 'random':
                self.train_ind, self.test_ind = DSprites.get_partition(range_all, range_test)
            else:
                all_ind = DSprites.get_partition(range_all)
                # shuffled_ids_cache_path = os.path.join(self.data_dir, f"{self.name}_shuffled_ids.npy")
                shuffled_ids_cache_path = os.path.join(self.data_dir, f"{self.class_name}_{self.mode}_seed{self.random_seed}_shuffled_ids.npy")

                if os.path.isfile(shuffled_ids_cache_path):
                    print(f"Load shuffled ids at {shuffled_ids_cache_path}")
                    shuffled_ids = np.load(shuffled_ids_cache_path)
                else:
                    print(f"Save shuffled ids at {shuffled_ids_cache_path}")
                    shuffled_ids = np.random.permutation(all_ind)
                    np.save(shuffled_ids_cache_path, shuffled_ids)

                self.train_ind = shuffled_ids[test_size:]
                self.test_ind = shuffled_ids[:test_size]

            if self.n_train is not None:
                self.train_ind = self.train_ind[:int(self.n_train*self.n_fold)]

        elif self.class_name == 'multidsprites':
            training_size = int(0.6 * MultiDsprites.NUM_SAMPLES)
            shuffled_ids_cache_path = os.path.join(self.data_dir, f"{self.name}_shuffled_ids_seed{self.random_seed}.npy")
            if os.path.isfile(shuffled_ids_cache_path):
                print("Load shuffled ids")
                shuffled_ids = np.load(shuffled_ids_cache_path)
            else:
                shuffled_ids = np.random.permutation(int(MultiDsprites.NUM_SAMPLES))
                print("Save shuffled ids")
                np.save(shuffled_ids_cache_path, shuffled_ids)
            self.train_ind = shuffled_ids[:training_size]
            self.test_ind = shuffled_ids[training_size:]

            # for only use part of training set
            if self.n_train is not None:
                self.train_ind = self.train_ind[:self.n_train]
        else:
            raise ValueError('Undefined dataset type')


    def setup(self, *args, **kwargs):
        self.train_dataset = self.dataset_class(root=self.data_dir,
                                      range=self.train_ind,
                                      transform=self.transform,
                                      n_samples=self.virtual_n_samples,
                                      **self.dataset_config
                                      )
        self.test_dataset = self.dataset_class(root=self.data_dir,
                                     range=self.test_ind,
                                     transform=self.transform,
                                     **self.dataset_config
                                     )


class MPI3DDataModule(MetaDataModule):
    """
    Supported name format : "mpi3d_{subset}_{split_mode}_v{version}"
    subset = {toy, realistic, real}
    split_mode = {random}
    version = {}
    """
    def prepare_data(self) -> None:
        self.subset_name = self.name.split('_')[1].lower()
        self.mode = self.name.split('_')[2].lower()
        self.version = self.name.split('_')[3].lower()

        if self.mode == 'random':
            # total size 1036800
            test_ratio = {
                'v1': 1/6,  # 5: 1
                'v2': 1/3,  # 2: 1
                'v3': 1/2,  # 1: 1
                'v4': 7/10,  # 3: 7
                'v5': 9/10,  # 1: 9
                'v6': 95/100,  # 5: 95
                'v7': 99/100,  # 1: 99
            }
            test_size = int(self.dataset_class.total_sample_size * test_ratio[self.version])

            all_ind = list(np.arange(self.dataset_class.total_sample_size))
            shuffled_ids_cache_path = os.path.join(self.data_dir, f"{self.class_name}_{self.subset_name}_{self.mode}_shuffled_ids_seed{self.random_seed}.npy")
            if os.path.isfile(shuffled_ids_cache_path):
                print(f"Load shuffled ids at {shuffled_ids_cache_path}")
                shuffled_ids = np.load(shuffled_ids_cache_path)
            else:
                print(f"Save shuffled ids at {shuffled_ids_cache_path}")
                shuffled_ids = np.random.permutation(all_ind)
                np.save(shuffled_ids_cache_path, shuffled_ids)

            self.train_ind = shuffled_ids[test_size:]
            self.test_ind = shuffled_ids[:test_size]

        if self.n_train is not None:
            self.train_ind = self.train_ind[:int(self.n_train*self.n_fold)]

    def setup(self, *args, **kwargs):
        self.train_dataset = self.dataset_class(root=self.data_dir,
                                                range=self.train_ind,
                                                subset=self.subset_name,
                                                n_samples=self.virtual_n_samples,
                                                )
        self.test_dataset = self.dataset_class(root=self.data_dir,
                                               range=self.test_ind,
                                               subset=self.subset_name,
                                               )