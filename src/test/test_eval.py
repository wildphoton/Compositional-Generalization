#!/usr/bin/env python
"""
Created by zhenlinx on 02/25/2022
"""
import os
import sys
sys.path.append(os.path.realpath('..'))

import torch
from pytorch_lightning import seed_everything
from datasets import get_datamodule
from evaluation import DCIMetrics
from evaluation import CompGenalizationMetrics, ScikitLearnEvaluator
from evaluation import DisentangleMetricEvaluator
from evaluation import TopoSimEval
from models import VAE, RecurrentDiscreteVAE


def testDisentangleMetrics():
    model = VAE(
        input_size=(1, 64, 64),
        architecture='burgess',
        latent_size=8,
        beta=1,
    )
    ckpoint_path = '/playpen-raid2/zhenlinx/Code/discrete_comp_gen/src/scripts/logs/dsprites90d_random_v1/VAE_burgess_z10_beta0.5_bce_lr0.0001_adam_wd0_dsprites90d_random_v1_batch64_100epochs/version_2001/checkpoints/last.ckpt'
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
    metric = DCIMetrics(dm.train_dataloader(), n_factors=dm.train_dataset.num_factors)

    results = metric(model)
    print(results[2])
    pass


def testCompGenalizationEval():
    model = VAE(
        input_size=(3, 64, 64),
        architecture='burgess',
        latent_size=10,
        beta=0,
    )
    # ckpoint_path = '/playpen-raid2/zhenlinx/Code/discrete_comp_gen/src/scripts/logs/dsprites90d_random_v1/VAE_burgess_z10_beta0.5_bce_lr0.0001_adam_wd0_dsprites90d_random_v1_batch64_100epochs/version_2001/checkpoints/last.ckpt'
    # ckpt = torch.load(ckpoint_path, map_location=torch.device('cpu'))
    # model.load_state_dict(ckpt['state_dict'])
    model = model.to('cuda:0')
    n_train = 200
    # for mode in ('pre', 'latent', 'post'):
    for mode in ('post', ):
        dm = get_datamodule('mpi3d_real_random_v1',
                            data_dir="/playpen-raid2/zhenlinx/Data/disentanglement/mpi3d",
                            batch_size=64,
                            num_workers=0,
                            n_train=n_train,
                            use_latent_class=True
                            )
        dm.prepare_data()
        dm.setup()
        task_types = dm.train_dataset.task_types
        cls_metric = CompGenalizationMetrics(dm.train_dataloader(),
                                             dm.test_dataloader(),
                                             n_factors=dm.train_dataset.num_factors,
                                             model='logistic',
                                             factor_selector=(task_types == 'cls').nonzero()[0]
                                             # max_iter=1000,
                                             # normalize=True,
                                             )
        results_cls = cls_metric(model, mode=mode)
        print(f'{mode} cls', results_cls, results_cls.mean())

        dm = get_datamodule('mpi3d_real_random_v1',
                            data_dir="/playpen-raid2/zhenlinx/Data/disentanglement/mpi3d",
                            batch_size=64,
                            num_workers=0,
                            n_train=n_train,
                            use_latent_class=False
                            )
        dm.prepare_data()
        dm.setup()
        reg_metric = CompGenalizationMetrics(dm.train_dataloader(),
                                             dm.test_dataloader(),
                                             n_factors=dm.train_dataset.num_factors,
                                             model='linear',
                                             factor_selector=(task_types == 'reg').nonzero()[0]
                                             )

        results_reg = reg_metric(model, mode=mode)
        print(f'{mode} cls', results_cls, results_cls.mean())
        print(f'{mode} reg', results_reg, results_reg.mean())
    pass

def testScikit_learn_evaluator():
    seed = 2002
    seed_everything(seed)
    model = VAE(
        input_size=(1, 64, 64),
        architecture='base',
        latent_size=10,
        beta=0,
    )
    # ckpoint_path = f'/playpen-raid2/zhenlinx/Code/discrete_comp_gen/src/scripts/logs/dsprites90d_random_v6/VAE_base_z10_beta0_bce_lr0.0001_adam_wd0_dsprites90d_random_v6_batch64_100epochs/version_{seed}/checkpoints/last.ckpt'
    ckpoint_path = f'/playpen-raid2/zhenlinx/Code/discrete_comp_gen/src/scripts/logs/dsprites90d_random_v5/VAE_base_z10_beta0_bce_lr0.0001_adam_wd0_dsprites90d_random_v5_batch64_100epochs/version_{seed}/checkpoints/last.ckpt'

    # model = RecurrentDiscreteVAE(
    #     input_size=(1, 64, 64),
    #     architecture='base',
    #     latent_size=10,
    #     dictionary_size=256,
    #     beta=0,
    # )
    # ckpoint_path = f'/playpen-raid2/zhenlinx/Code/discrete_comp_gen/src/scripts/logs/dsprites90d_random_v6/RecurrentDiscreteVAE_base_z10_D256_gsmT1_beta0_bce_lr0.0001_adam_wd0_dsprites90d_random_v6_batch64_100epochs/version_{seed}/checkpoints/last.ckpt'

    ckpt = torch.load(ckpoint_path, map_location=torch.device('cpu'))
    model.load_state_dict(ckpt['state_dict'])
    model = model.to('cuda:0')
    n_train = 1000
    n_fold = 1
    # dm = get_datamodule('mpi3d_real_random_v1',
    #                     data_dir="/playpen-raid2/zhenlinx/Data/disentanglement/mpi3d",
    #                     batch_size=64,
    #                     num_workers=0,
    #                     n_train=n_train,
    #                     n_fold=n_fold,
    #                     random_seed=2001,
    #                     )
    dm = get_datamodule('dsprites90d_random_v5',
                        data_dir="/playpen-raid2/zhenlinx/Data/disentanglement/dsprites",
                        batch_size=64,
                        num_workers=0,
                        n_train=n_train,
                        n_fold=n_fold,
                        random_seed=seed,
                        )
    # for mode in ('pre', 'latent', 'post'):
    for mode in ('post', ):
        evaluator = ScikitLearnEvaluator(
            backbone=model,
            datamodule=dm,
            mode=mode,
            n_train=n_train,
            n_fold=n_fold,
            reverse_task_type=False,
            reg_model='GBTR',
            cls_model='GBTC',
        )
        print(evaluator.name)
        results = evaluator.eval()
        print(results)

def test_disentangle_score_eval():
    seed = 2002
    seed_everything(seed)
    model = VAE(
        input_size=(3, 64, 64),
        architecture='base',
        latent_size=10,
        beta=0,
    )
    # ckpoint_path = f'/playpen-raid2/zhenlinx/Code/discrete_comp_gen/src/scripts/logs/dsprites90d_random_v6/VAE_base_z10_beta0_bce_lr0.0001_adam_wd0_dsprites90d_random_v6_batch64_100epochs/version_{seed}/checkpoints/last.ckpt'
    # ckpoint_path = f'/playpen-raid2/zhenlinx/Code/discrete_comp_gen/src/scripts/logs/dsprites90d_random_v5/VAE_base_z10_beta8_bce_lr0.0001_adam_wd0_dsprites90d_random_v5_batch64_100epochs/version_{seed}/checkpoints/last.ckpt'
    ckpoint_path = f'/playpen-raid2/zhenlinx/Code/discrete_comp_gen/src/scripts/logs/mpi3d_real_random_v5/VAE_base_z10_beta8_bce_lr0.0001_adam_wd0_mpi3d_real_random_v5_batch64_100epochs/version_{seed}/checkpoints/last.ckpt'
    ckpt = torch.load(ckpoint_path, map_location=torch.device('cpu'))
    model.load_state_dict(ckpt['state_dict'])
    model = model.to('cuda:0')
    n_train = 10000
    n_fold = 1
    # dm = get_datamodule(name='dsprites90d_random_v5',
    #                     data_dir="/playpen-raid2/zhenlinx/Data/disentanglement/dsprites",
    #                     batch_size=64,
    #                     num_workers=0,
    #                     # n_train=n_train,
    #                     # n_fold=n_fold,
    #                     random_seed=seed,
    #                     )
    dm = get_datamodule('mpi3d_real_random_v5',
                        data_dir="/playpen-raid2/zhenlinx/Data/disentanglement/mpi3d",
                        batch_size=64,
                        num_workers=0,
                        # n_train=n_train,
                        # n_fold=n_fold,
                        random_seed=2001,
                        )
    disentanglemetrics = DisentangleMetricEvaluator(
        model=model,
        datamodule=dm
    )
    res = disentanglemetrics.eval()
    print(res)

def test_topsim_eval():
    seed = 2002
    seed_everything(seed)
    model = RecurrentDiscreteVAE(
        input_size=(1, 64, 64),
        architecture='base',
        latent_size=10,
        dictionary_size=256,
        beta=0,
    )
    ckpoint_path = f'/playpen-raid2/zhenlinx/Code/discrete_comp_gen/src/scripts/logs/dsprites90d_random_v5/RecurrentDiscreteVAE_base_z10_D256_gsmT1_beta0_bce_lr0.0001_adam_wd0_dsprites90d_random_v5_batch64_100epochs/version_{seed}/checkpoints/last.ckpt'
    ckpt = torch.load(ckpoint_path, map_location=torch.device('cpu'))
    model.load_state_dict(ckpt['state_dict'])
    model = model.to('cuda:0')
    dm = get_datamodule('dsprites90d_random_v5',
                        data_dir="/playpen-raid2/zhenlinx/Data/disentanglement/dsprites",
                        batch_size=64,
                        num_workers=0,
                        # n_train=1,
                        # n_fold=n_fold,
                        random_seed=2001,
                        )
    comp_metric = TopoSimEval(
        model=model,
        datamodule=dm
    )
    res = comp_metric.eval()
    print(res)

if __name__ == '__main__':
    # testDisentangleMetrics()
    # testCompGenalizationEval()
    # testScikit_learn_evaluator()
    # test_disentangle_score_eval()
    test_topsim_eval()

# # {'R2_scale': array([0.50475683, 0.3401749 , 0.28559243]), 'R2_orient': array([-1.48719895, -1.2153346 , -3.07160444]), 'R2_posX': array([0.95714793, 0.94912344, 0.94077351]), 'R2_posY': array([0.95082625, 0.93245295, 0.94706246]), 'R2_mean': 0.08614772630398974, 'acc_shape': array([0.69964707, 0.70638021, 0.73080569]), 'acc_mean': 0.7122776559454191}
# Rev {'R2_shape': 0.8572552929899554, 'R2_mean': 0.8572552929899554, 'acc_scale': 0.9227068865740741, 'acc_orient': 0.49576220100308643, 'acc_posX': 0.37700135030864196, 'acc_posY': 0.38262562692901236, 'acc_mean
# ': 0.5445240162037037}

# 1fold {'R2_scale': 0.9545822973977659, 'R2_orient': 0.4621714901111502, 'R2_posX': 0.9893586725771996, 'R2_posY': 0.9865271063729859, 'R2_mean': 0.8481598916147755, 'acc_shape': 0.9910300925925926, 'acc_mean': 0.9910300925925926}
#10fold {'R2_scale': 0.954311342804011, 'R2_orient': 0.46364597811848596, 'R2_posX': 0.9891487202507033, 'R2_posY': 0.9866062595192636, 'R2_mean': 0.8484280751731159, 'acc_shape': 0.9894428771219136, 'acc_mean': 0.9894428771219136}
# 1fold reset {'R2_scale': 0.9545769953027611, 'R2_orient': 0.46222533069249233, 'R2_posX': 0.9893567930814057, 'R2_posY': 0.9865262293930906, 'R2_mean': 0.8481713371174374, 'acc_shape': 0.9910059799382716, 'acc_mean': 0.9910059799382716}