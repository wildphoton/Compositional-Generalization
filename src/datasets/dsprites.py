#!/usr/bin/env python
"""
Copied from https://github.com/YannDubs/disentangling-vae/blob/master/utils/datasets.py
Created by zhenlinx on 02/24/2021
"""
import abc
import logging
import os
import subprocess

# from skimage.io import imread
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

DIR = os.path.abspath(os.path.dirname(__file__))
COLOUR_BLACK = 0
COLOUR_WHITE = 1
DATASETS_DICT = {
    # "mnist": "MNIST",
    #                  "fashion": "FashionMNIST",
    "dsprites": "DSprites",
    # "celeba": "CelebA",
    # "chairs": "Chairs"
}
DATASETS = list(DATASETS_DICT.keys())


def get_dataset(dataset):
    """Return the correct dataset."""
    dataset = dataset.lower()
    try:
        # eval because stores name as string in order to put it at top of file
        return eval(DATASETS_DICT[dataset])
    except KeyError:
        raise ValueError("Unkown dataset: {}".format(dataset))


def get_img_size(dataset):
    """Return the correct image size."""
    return get_dataset(dataset).img_size


def get_background(dataset):
    """Return the image background color."""
    return get_dataset(dataset).background_color


def get_dataloaders(dataset, root=None, shuffle=True, pin_memory=True,
                    batch_size=128, logger=logging.getLogger(__name__), **kwargs):
    """A generic data loader

    Parameters
    ----------
    dataset : {"mnist", "fashion", "dsprites", "celeba", "chairs"}
        Name of the dataset to load

    root : str
        Path to the dataset root. If `None` uses the default one.

    kwargs :
        Additional arguments to `DataLoader`. Default values are modified.
    """
    pin_memory = pin_memory and torch.cuda.is_available  # only pin if GPU available
    Dataset = get_dataset(dataset)
    dataset = Dataset(logger=logger) if root is None else Dataset(root=root, logger=logger)
    return DataLoader(dataset,
                      batch_size=batch_size,
                      shuffle=shuffle,
                      pin_memory=pin_memory,
                      **kwargs)


class DisentangledDataset(Dataset, abc.ABC):
    """Base Class for disentangled VAE datasets.

    Parameters
    ----------
    root : string
        Root directory of dataset.

    transforms_list : list
        List of `torch.vision.transforms` to apply to the data when loading it.
    """

    def __init__(self, root, logger=logging.getLogger(__name__)):
        self.root = root
        self.train_data = os.path.join(root, type(self).files["train"])
        # self.transforms = transforms.Compose(transforms_list)
        self.logger = logger

        if not os.path.isdir(root):
            self.logger.info("Downloading {} ...".format(str(type(self))))
            self.download()
            self.logger.info("Finished Downloading.")

    def __len__(self):
        return len(self.imgs)

    @abc.abstractmethod
    def __getitem__(self, idx):
        """Get the image of `idx`.

        Return
        ------
        sample : torch.Tensor
            Tensor in [0.,1.] of shape `img_size`.
        """
        pass

    @abc.abstractmethod
    def download(self):
        """Download the dataset. """
        pass


class DSprites(DisentangledDataset):
    """DSprites Dataset from [1].

    Disentanglement test Sprites dataset.Procedurally generated 2D shapes, from 6
    disentangled latent factors. This dataset uses 6 latents, controlling the color,
    shape, scale, rotation and position of a sprite. All possible variations of
    the latents are present. Ordering along dimension 1 is fixed and can be mapped
    back to the exact latent values that generated that image. Pixel outputs are
    different. No noise added.

    Notes
    -----
    - Link : https://github.com/deepmind/dsprites-dataset/
    - hard coded metadata because issue with python 3 loading of python 2

    Parameters
    ----------
    root : string
        Root directory of dataset.

    References
    ----------
    [1] Higgins, I., Matthey, L., Pal, A., Burgess, C., Glorot, X., Botvinick,
        M., ... & Lerchner, A. (2017). beta-vae: Learning basic visual concepts
        with a constrained variational framework. In International Conference
        on Learning Representations.

    """
    urls = {
        "train": "https://github.com/deepmind/dsprites-dataset/blob/master/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz?raw=true"}
    files = {"train": "dsprite_train.npz"}
    lat_names = ('shape', 'scale', 'orientation', 'posX', 'posY')
    lat_sizes = np.array([3, 6, 40, 32, 32])
    NUM_CLASSES = list(lat_sizes)
    img_size = (1, 64, 64)
    n_gen_factors = 5
    total_sample_size = 737280
    background_color = COLOUR_BLACK
    latents_values = {'color': np.array([1.]),
                      'shape': np.array([1., 2., 3.]),  # square, ellipse, heart
                      'scale': np.array([0.5, 0.6, 0.7, 0.8, 0.9, 1.]),
                      'orientation': np.array([0., 0.16110732, 0.32221463, 0.48332195,
                                               0.64442926, 0.80553658, 0.96664389, 1.12775121,
                                               1.28885852, 1.44996584, 1.61107316, 1.77218047,
                                               1.93328779, 2.0943951, 2.25550242, 2.41660973,
                                               2.57771705, 2.73882436, 2.89993168, 3.061039,
                                               3.22214631, 3.38325363, 3.54436094, 3.70546826,
                                               3.86657557, 4.02768289, 4.1887902, 4.34989752,
                                               4.51100484, 4.67211215, 4.83321947, 4.99432678,
                                               5.1554341, 5.31654141, 5.47764873, 5.63875604,
                                               5.79986336, 5.96097068, 6.12207799, 6.28318531]),  # [0, 2 pi]
                      'posX': np.array([0., 0.03225806, 0.06451613, 0.09677419, 0.12903226,
                                        0.16129032, 0.19354839, 0.22580645, 0.25806452,
                                        0.29032258, 0.32258065, 0.35483871, 0.38709677,
                                        0.41935484, 0.4516129, 0.48387097, 0.51612903,
                                        0.5483871, 0.58064516, 0.61290323, 0.64516129,
                                        0.67741935, 0.70967742, 0.74193548, 0.77419355,
                                        0.80645161, 0.83870968, 0.87096774, 0.90322581,
                                        0.93548387, 0.96774194, 1.]),
                      'posY': np.array([0., 0.03225806, 0.06451613, 0.09677419, 0.12903226,
                                        0.16129032, 0.19354839, 0.22580645, 0.25806452,
                                        0.29032258, 0.32258065, 0.35483871, 0.38709677,
                                        0.41935484, 0.4516129, 0.48387097, 0.51612903,
                                        0.5483871, 0.58064516, 0.61290323, 0.64516129,
                                        0.67741935, 0.70967742, 0.74193548, 0.77419355,
                                        0.80645161, 0.83870968, 0.87096774, 0.90322581,
                                        0.93548387, 0.96774194, 1.]),
                      }

    def __init__(self, root, range, use_latent_class=True, transform=None, n_samples=None, **kwargs):
        """

        :param root:
        :param range: an array of indices for a subset of the data
        :param laten_class: return latent variables as classes instead of values
        :param transform:
        :param kwargs:
        """
        super().__init__(root, **kwargs)

        dataset_zip = np.load(self.train_data, allow_pickle=True, encoding='latin1')
        self.meta_data = dataset_zip['metadata'][()]
        self.use_latent_class = use_latent_class

        self.imgs = dataset_zip['imgs']
        self.latents_values = dataset_zip['latents_values'][:, 1:]
        self.latents_classes = dataset_zip['latents_classes'][:, 1:]

        self.imgs = self.imgs[range]
        self.latents_values = self.latents_values[range]
        self.latents_classes = self.latents_classes[range]

        self.imgs = torch.from_numpy(self.imgs).unsqueeze(1).float()
        self.latents_values = torch.from_numpy(self.latents_values).float()
        self.latents_classes = torch.from_numpy(self.latents_classes).long()

        self.transform = transform
        self.n_samples = n_samples if n_samples is not None else len(self.imgs)
        self.raw_num_samples = len(self.imgs)
        # self.imgs = (self.imgs - 0.5)/0.5

    def download(self):
        """Download the dataset."""
        os.makedirs(self.root)
        subprocess.check_call(["curl", "-L", type(self).urls["train"],
                               "--output", self.train_data])

    def __getitem__(self, idx):
        """Get the image of `idx`
        Return
        ------
        sample : torch.Tensor
            Tensor in [0.,1.] of shape `img_size`.

        lat_value : np.array
            Array of length 6, that gives the value of each factor of variation.
        """
        # map the recursive id to real id
        idx = idx % self.raw_num_samples

        # stored image have binary and shape (H x W) so multiply by 255 to get pixel
        # values + add dimension
        sample = self.imgs[idx]

        latent = self.latents_classes[idx] if self.use_latent_class else self.latents_values[idx]
        # lat_cls = self.latents_classes[idx]
        # lat_value = self.latents_values[idx]
        if self.transform is not None:
            sample = self.transform(sample)
            # lat_value = self.transform(lat_value)
        return sample, latent

    def __len__(self):
        return self.n_samples

    # def map_cls_to_val(self, cls):
    #     """
    #     map class labels to actually values
    #     :param cls: batch of class labels for 5 channels
    #     :return: latent values
    #     """

    @staticmethod
    def get_partition(range_all: list, range_test=None):
        """
        Get a data partition given the range of each factor.
        :param range: a list of 5 arrays (color, shape, scale, rotation, pos_x, pos_y)
            each array is a range of indices for the corresponding properties
            e.g. [[0, ], [1,], [0, 1], np.arange(13,26), np.arange(21, 31), np.arange(21, 31)].
            ** range_test could also be a tuple of such lists for merge multiple possible ranges
        :return: the corresponding indices of the partition data.
            when range_test is None, return indice for range_all;
            when range_test is given, return indices for range_all - range_test, and for range_test
        """

        all_indices = DSprites.range_to_index(range_all)
        if range_test is None:
            return all_indices
        else:
            test_indices = DSprites.range_to_index(range_test)
            assert len(np.setdiff1d(test_indices, all_indices)) == 0  # check if test indices is a subset of all indices
            train_indices = np.setdiff1d(all_indices, test_indices)
            return train_indices, test_indices

    @staticmethod
    def range_to_index(latent_range):
        if latent_range is None:
            return np.arange(0, DSprites.total_sample_size)
        else:
            if type(latent_range) is list:
                latents_sampled = np.array(np.meshgrid(*latent_range)).T.reshape(-1, len(latent_range))
                indices_sampled = DSprites.latent_to_index(latents_sampled)
            elif type(latent_range) is tuple:
                multiple_indices_sampled = [DSprites.latent_to_index(
                    np.array(np.meshgrid(*range)).T.reshape(-1, len(range))) for range in latent_range]
                indices_sampled = np.concatenate(multiple_indices_sampled, axis=0)
            else:
                raise ValueError()
            return indices_sampled

    @staticmethod
    def latent_to_index(latents):
        latents_sizes = DSprites.lat_sizes
        latents_bases = np.concatenate((latents_sizes[::-1].cumprod()[::-1][1:],
                                        np.array([1, ])))

        return np.dot(latents, latents_bases).astype(int)

    # def __len__(self):
    #     return 1000

# if __name__ == '__main__':
# transform = transforms.Normalize((0.5,), (0.5,))
# range_all = [np.arange(3), np.arange(6), np.arange(10), np.arange(32), np.arange(32)]
# range_test = [[1, ], [0, 1], np.arange(6, 10), np.arange(21, 32), np.arange(21, 32)]
# dataset = DSprites(root=os.path.join(DIR, '../data/dsprites/'),
#                    range_all=None,
#                    range_test=None,
#                    train=True,
#                    transform=None
#                    )
# dm = DSpritesDataModule(name='dsprites90d_random_v4', data_dir='../data/dsprites',
#                         batch_size = 128, num_workers=0, n_train=None)
# dm.prepare_data()
# dm.setup()
# pass
