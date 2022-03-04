# import unittest
from datasets.helper import get_datamodule
from tqdm import tqdm

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
    dm = get_datamodule(name='dsprites_all', data_dir='../data/dsprites',
                            batch_size = 64, num_workers=4, n_train=None)
    print(dm.num_classes)
    dm.prepare_data()
    dm.setup()
    print(len(dm.train_dataset))
    for data in tqdm(dm.train_dataloader()):
        pass


if __name__ == '__main__':
    # unittest.main()
    test_dsprites_module()