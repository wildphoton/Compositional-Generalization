#!/usr/bin/env python
"""
Created by zhenlinx on 02/25/2022
"""
import os
import sys
sys.path.append(os.path.realpath('..'))

import torch
from datasets import get_datamodule
from evaluation import DCIMetrics, CompGenalizationMetrics
from models.vae import VAE

def testDisentangleMetrics():
    model = VAE(
        input_size=(1, 64, 64),
        architecture='burgess',
        latent_dim=8,
        beta=1,
    )
    ckpoint_path = '/playpen-raid2/zhenlinx/Code/discrete_comp_gen/src/scripts/logs/dsprites90d_random_v1/VAE_burgess_z10_beta1_bce_lr0.0001_adam_wd0_dsprites90d_random_v1_batch64_100epochs/version_2001/checkpoints/last.ckpt'
    ckpt = torch.load(ckpoint_path, map_location=torch.device('cpu'))
    model.load_state_dict(ckpt['state_dict'])

    dm = get_datamodule('dsprites90d_random_v1',
                        data_dir="/playpen-raid2/zhenlinx/Data/disentanglement/dsprites",
                        batch_size=64,
                        num_workers=0,
                        n_train=None,
                        use_latent_class=False
                        )
    dm.prepare_data()
    dm.setup()
    metric = DCIMetrics(dm.train_dataloader(), n_factors=dm.train_dataset.n_gen_factors)

    results = metric(model)
    print(results[2])
    pass


def testCompGenalizationEval():
    model = VAE(
        input_size=(1, 64, 64),
        architecture='burgess',
        latent_dim=10,
        beta=0,
    )
    ckpoint_path = '/playpen-raid2/zhenlinx/Code/discrete_comp_gen/src/scripts/logs/dsprites90d_random_v1/VAE_burgess_z10_beta1_bce_lr0.0001_adam_wd0_dsprites90d_random_v1_batch64_100epochs/version_2001/checkpoints/last.ckpt'
    ckpt = torch.load(ckpoint_path, map_location=torch.device('cpu'))
    model.load_state_dict(ckpt['state_dict'])
    model = model.to('cuda:0')
    n_train = 1000
    for mode in ('pre', 'latent', 'post'):
    # for mode in ('post', ):
    #     dm = get_datamodule('dsprites90d_random_v1',
    #                         data_dir="/playpen-raid2/zhenlinx/Data/disentanglement/dsprites",
    #                         batch_size=64,
    #                         num_workers=2,
    #                         n_train=n_train,
    #                         use_latent_class=True
    #                         )
    #     dm.prepare_data()
    #     dm.setup()
        # cls_metric = CompGenalizationMetrics(dm.train_dataloader(),
        #                                  dm.test_dataloader(),
        #                                  n_factors=dm.train_dataset.n_gen_factors,
        #                                  regressor='logistic',
        #                                  # normalize=True,
        #                                  )
        # results_cls = cls_metric(model, mode=mode)
        # print(f'{mode} cls', results_cls, results_cls.mean())

        dm = get_datamodule('dsprites90d_random_v1',
                            data_dir="/playpen-raid2/zhenlinx/Data/disentanglement/dsprites",
                            batch_size=64,
                            num_workers=2,
                            n_train=n_train,
                            use_latent_class=False
                            )
        dm.prepare_data()
        dm.setup()
        reg_metric = CompGenalizationMetrics(dm.train_dataloader(),
                                         dm.test_dataloader(),
                                         n_factors=dm.train_dataset.n_gen_factors,
                                         regressor='ridge',
                                         # normalize=True,
                                         )

        results_reg = reg_metric(model, mode=mode)
        print(f'{mode} reg', results_reg, results_reg.mean())
    pass

if __name__ == '__main__':
    # testDisentangleMetrics()
    testCompGenalizationEval()
