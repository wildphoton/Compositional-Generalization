# import unittest
import os
import sys
sys.path.append(os.path.realpath('..'))

from datasets.helper import get_datamodule
from tqdm import tqdm
from datasets import DSprites, MPI3D
import numpy as np
# class MyTestCase(unittest.TestCase):
    # def test_multisprites_module(self):
    #     dm = get_datamodule('multidsprites',
    #                         data_dir="/playpen-raid2/zhenlinx/Data/multi-objects/multi-dsprites",
    #                         batch_size=32,
    #                         num_workers=1,
    #                         )
    #     print(dm.num_classes)
    #     dm.prepare_data()
    #     dm.setup()
    #     print(len(dm.train_dataset))
    #     print(len(dm.train_dataloader))

def test_dsprites_module():
    dm = get_datamodule(name='dsprites_all', data_dir='/playpen-raid2/zhenlinx/Data/disentanglement/dsprites',
                            batch_size = 64, num_workers=4, n_train=None)
    print(dm.num_classes)
    dm.prepare_data()
    dm.setup()
    print(len(dm.train_dataset))
    for data in tqdm(dm.train_dataloader()):
        pass

def test_mpi3d_module():
    dm = get_datamodule(name='mpi3d_real_random_v1', data_dir='/playpen-raid2/zhenlinx/Data/disentanglement/mpi3d',
                        batch_size=64, num_workers=4, n_train=None)
    print(dm.num_classes)
    dm.prepare_data()
    dm.setup()
    print(len(dm.train_dataset))
    for data in tqdm(dm.train_dataloader()):
        pass

def test_dsprites_split_comp_ratio():
    range_all = [np.arange(3), np.arange(6), np.arange(10), np.arange(32), np.arange(32)]
    all_ind = DSprites.get_partition(range_all)
    shuffled_ids = np.random.permutation(all_ind)
    dataset = DSprites(root='/playpen-raid2/zhenlinx/Data/disentanglement/dsprites',
                      range=shuffled_ids,
                      )
    ys = dataset.latents_classes
    for ratio in (0.1, ):
        n_train = int(len(dataset)*ratio)
        print(n_train)
        train_ys = ys[:n_train]
        for k in range(dataset.n_gen_factors):
            lat_size = len(range_all[k])
            unique_val_k, unique_count_k = train_ys[:, k].unique(return_counts=True)
            n_unique_k = unique_val_k.shape[0]
            print(lat_size==n_unique_k, unique_val_k, unique_count_k)

def test_mpi3d_split_comp_ratio():
    all_ind = list(np.arange(MPI3D.total_sample_size))
    shuffled_ids = np.random.permutation(all_ind)
    dataset = MPI3D(root='/playpen-raid2/zhenlinx/Data/disentanglement/mpi3d',
                    subset='real',
                   range=shuffled_ids,
                   )
    ys = dataset.latent_values
    for ratio in (0.1,):
        n_train = int(len(dataset) * ratio)
        print(n_train)
        train_ys = ys[:n_train]
        for k in range(dataset.n_gen_factors):
            lat_size = dataset.lat_sizes[k]
            unique_val_k, unique_count_k = train_ys[:, k].unique(return_counts=True)
            n_unique_k = unique_val_k.shape[0]
            print(lat_size == n_unique_k, unique_val_k, unique_count_k)
    pass


if __name__ == '__main__':
    # unittest.main()
    # test_dsprites_module()
    # test_mpi3d_module()
    test_dsprites_split_comp_ratio()
    test_mpi3d_split_comp_ratio()