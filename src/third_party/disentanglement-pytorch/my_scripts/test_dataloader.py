#!/usr/bin/env python
"""
Created by zhenlinx on 02/18/2022
"""
import sys
from common.data_loader import get_dataloader
from common.arguments import get_args
from tqdm import tqdm

def main(args):
    loader = get_dataloader(args.dset_name, args.dset_dir, args.batch_size, args.seed, args.num_workers,
                   args.image_size, args.include_labels, args.pin_memory, not args.test,
                   not args.test)
    for data in tqdm(loader):
        pass


if __name__ == '__main__':
    _args = get_args(sys.argv[1:])
    main(_args)
