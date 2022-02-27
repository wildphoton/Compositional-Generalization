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


class DSpritesDataModule(pl.LightningDataModule):

    def __init__(self, name, data_dir, batch_size: int = 128, num_workers=4, n_train=None, virtual_n_samples=None, **dataset_config):
        super(DSpritesDataModule, self).__init__()
        self.name = name
        self.data_dir = data_dir
        self.batch_size = batch_size
        # self.transform = transforms.Normalize((0.5,), (0.5,))
        self.transform = None
        self.num_workers = num_workers
        self.n_train = n_train # use only partial of training set
        self.virtual_n_samples = virtual_n_samples # change the number of samples to be total training steps for prefetching purpose
        if self.name.split('_')[0] == 'multidsprites':
            self.dataset_class = MultiDsprites
        else:
            self.dataset_class = DSprites
        self.num_classes = self.dataset_class.NUM_CLASSES # unique classes for each attribute
        self.dataset_config = dataset_config

    def prepare_data(self):
        if self.name.split('_')[0] == 'dsprites':

            if self.name.split('_')[1] == 'element':
                range_test = [[1, ], [0, 1], np.arange(13, 26), np.arange(21, 32), np.arange(21, 32)]
            elif self.name.split('_')[1] == 'range':
                range_test = [[0, ], np.arange(6), np.arange(40), np.arange(16, 32), np.arange(32)]
            elif self.name.split('_')[1] == 'all':
                range_test = None
            else:
                raise ValueError('Undefined splitting')

            range_all = [np.arange(3), np.arange(6), np.arange(40), np.arange(32), np.arange(32)]
            if range_test:
                self.train_ind, self.test_ind = DSprites.get_partition(range_all, range_test)
            else:
                self.train_ind = DSprites.get_partition(range_all, range_test)
                self.test_ind = self.train_ind

        elif self.name.split('_')[0] == 'dsprites90d':
            transform = transforms.Normalize((0.5,), (0.5,))
            range_all = [np.arange(3), np.arange(6), np.arange(10), np.arange(32), np.arange(32)]
            if self.name.split('_')[1] == 'element':
                if self.name.split('_')[2] == 'v1':
                    range_test = [[0, ], [0, 1], np.arange(6, 10), np.arange(21, 32), np.arange(21, 32)]
                elif self.name.split('_')[2] == 'v2':
                    range_test = [[1, ], [0, 1], np.arange(6, 10), np.arange(21, 32), np.arange(21, 32)]
                elif self.name.split('_')[2] == 'v3':
                    range_test = [[2, ], [0, 1], np.arange(6, 10), np.arange(21, 32), np.arange(21, 32)]
                else:
                    raise ValueError('Undefined splitting')

            elif self.name.split('_')[1] == 'multiElement':
                if self.name.split('_')[2] == 'v1':  # train/test: 179712/4608
                    range_test = (
                        [[0, ], [0, 1], np.arange(1, 4), np.arange(16, 32), np.arange(16, 32)],
                        [[1, ], [2, 3], np.arange(4, 7), np.arange(0, 16), np.arange(0, 16)],
                        [[2, ], [4, 5], np.arange(7, 10), np.arange(16, 32), np.arange(0, 16)],
                    )
                elif self.name.split('_')[2] == 'v2':  # train/test: 173568/10752
                    range_test = (
                        [[0, ], [0, 1, 2], np.arange(0, 5), np.arange(16, 32), np.arange(16, 32)],
                        [[1, ], [3, 4, 5], np.arange(6, 10), np.arange(0, 16), np.arange(0, 16)],
                        [[2, ], [1, 2, 3], np.arange(3, 8), np.arange(8, 24), np.arange(8, 24)],
                    )
                elif self.name.split('_')[2] == 'v3':  # 92160/92160
                    range_test = (
                        [[0, ], np.arange(6), np.arange(10), np.arange(16, 32), np.arange(32)],
                        [[1, ], np.arange(6), np.arange(10), np.arange(0, 16), np.arange(32)],
                        [[2, ], np.arange(6), np.arange(10), np.arange(8, 24), np.arange(32)]
                    )
                elif self.name.split('_')[2] == 'v4':  # 138240/46080
                    range_test = (
                        [[0, ], np.arange(6), np.arange(10), np.arange(16, 32), np.arange(8, 24)],
                        [[1, ], np.arange(6), np.arange(10), np.arange(0, 16), np.arange(16, 32)],
                        [[2, ], np.arange(6), np.arange(10), np.arange(8, 24), np.arange(0, 16)]
                    )
                elif self.name.split('_')[2] == 'v5':  # 156672/27648
                    range_test = (
                        [[0, ], np.arange(6), np.arange(1, 4), np.arange(16, 32), np.arange(32)],
                        [[1, ], np.arange(6), np.arange(4, 7), np.arange(0, 16), np.arange(32)],
                        [[2, ], np.arange(6), np.arange(7, 10), np.arange(8, 24), np.arange(32)]
                    )
                elif self.name.split('_')[2] == 'v6':  # 165888/18432
                    range_test = (
                        [[0, ], np.arange(0, 2), np.arange(1, 4), np.arange(32), np.arange(32)],
                        [[1, ], np.arange(2, 4), np.arange(4, 7), np.arange(32), np.arange(32)],
                        [[2, ], np.arange(4, 6), np.arange(7, 10), np.arange(32), np.arange(32)]
                    )
                else:
                    raise ValueError('Undefined splitting')
            elif self.name.split('_')[1] == 'range':
                # shape + posX
                if self.name.split('_')[2] == 'v1':
                    range_test = [[0, ], np.arange(6), np.arange(10), np.arange(16, 32), np.arange(32)]
                elif self.name.split('_')[2] == 'v2':
                    range_test = [[1, ], np.arange(6), np.arange(10), np.arange(16, 32), np.arange(32)]
                elif self.name.split('_')[2] == 'v3':
                    range_test = [[2, ], np.arange(6), np.arange(10), np.arange(16, 32), np.arange(32)]

                # scale + posX to check if shape has too low diversity
                elif self.name.split('_')[2] == 'v4':
                    range_test = [[0, 1, 2], np.arange(0, 2), np.arange(10), np.arange(16, 32), np.arange(32)]
                elif self.name.split('_')[2] == 'v5':
                    range_test = [[0, 1, 2], np.arange(2, 4), np.arange(10), np.arange(16, 32), np.arange(32)]
                elif self.name.split('_')[2] == 'v6':
                    range_test = [[0, 1, 2], np.arange(4, 6), np.arange(10), np.arange(16, 32), np.arange(32)]

                # orientation + posX to check if shape has too low diversity
                elif self.name.split('_')[2] == 'v7':
                    range_test = [[0, 1, 2], np.arange(6), np.arange(1, 4), np.arange(16, 32), np.arange(32)]
                elif self.name.split('_')[2] == 'v8':
                    range_test = [[0, 1, 2], np.arange(6), np.arange(4, 7), np.arange(16, 32), np.arange(32)]
                elif self.name.split('_')[2] == 'v9':
                    range_test = [[0, 1, 2], np.arange(6), np.arange(7, 10), np.arange(16, 32), np.arange(32)]


                else:
                    raise ValueError('Undefined version')
            elif self.name.split('_')[1] == 'random':
                # 184320 total images
                test_sizes = {
                    'v1': 30000,  # 5: 1
                    'v2': 60000,  # 2: 1
                    'v3': 90000,  # 1: 1
                }
                test_size = test_sizes[self.name.split('_')[2]]
                # total 184K
            else:
                raise ValueError('Undefined splitting')

            if self.name.split('_')[1] != 'random':
                self.train_ind, self.test_ind = DSprites.get_partition(range_all, range_test)
            else:
                all_ind = DSprites.get_partition(range_all)
                shuffled_ids_cache_path = os.path.join(self.data_dir, f"{self.name}_shuffled_ids.npy")
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
                self.train_ind = self.train_ind[:self.n_train]

        elif self.name.split('_')[0] == 'multidsprites':
            training_size = int(0.6 * MultiDsprites.NUM_SAMPLES)
            shuffled_ids_cache_path = os.path.join(self.data_dir, f"{self.name}_shuffled_ids.npy")
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

    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          batch_size=self.batch_size,
                          shuffle=True,
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


