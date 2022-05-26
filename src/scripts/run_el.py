#!/usr/bin/env python
"""
Created by zhenlinx on 03/30/2021
"""
import os
import sys
sys.path.append(os.path.realpath('..'))
import argparse
import yaml
from itertools import product
from scripts.experiments import add_vae_argument, train_vae, scikitlearn_eval, setup_experiment

def main():
    parser = argparse.ArgumentParser(description='Generic runner for VAE models')
    add_vae_argument(parser)
    args = parser.parse_args()
    args.filename = 'configs/rec_el.yaml'
    config, sklearn_eval_cfg = setup_experiment(args)

    # for data in ('dsprites90d_random_v5', ):
    for data in ('mpi3d_real_random_v6', ):
        for recon_loss, beta, latent_size, dict_size, arch in product(('bce', ),
                                                                      # (0, ), (8, 10, 12), (128, 256, 512),
                                                                      (0, ), (10, ), (512, 256),
                                                                      # ('burgess', 'burgess_wide')
                                                                      ('base', )
                                                                      # ('large', )
                                                                      ):
            for seed in (2001, 2002, 2003):
            # for seed in (2003, ):
                config['model_params']['name'] = 'RecurrentEL'
                config['model_params']['beta'] = beta
                config['model_params']['latent_size'] = latent_size
                config['model_params']['dictionary_size'] = dict_size
                config['model_params']['recon_loss'] = recon_loss
                config['model_params']['fix_length'] = False
                for determ in (False, ):
                    config['model_params']['deterministic'] = determ
                    config['model_params']['architecture'] = arch

                    config['exp_params']['random_seed'] = seed
                    # config['exp_params']['max_epochs'] = 200
                    # config['exp_params']['batch_size'] = 512
                    config['exp_params']['train_steps'] = 500000
                    config['exp_params']['val_steps'] = 5000

                    config['exp_params']['dataset'] = data
                    if 'mpi3d' in data:
                        config['exp_params'][
                            'data_path'] = '/playpen-raid2/zhenlinx/Data/disentanglement/mpi3d'
                        # config['exp_params']['max_epochs'] = 200  # 100 for dsprites and 50 for mpi3d
                        config['model_params']['input_size'] = [3, 64, 64]
                        config['exp_params']['train_steps'] = 1000000
                        config['exp_params']['val_steps'] = 10000

                    train_vae(config, args)

                    if args.sklearn:
                        # sklearn eval
                        # for mode, n_train in product(('post', ), (1000, 500, 100,), ):
                        # for mode, n_train in product(('pre', 'latent', 'post' ), (1000, 500, 100), ):
                        for mode, n_train in product(('post',  'pre', 'latent' ), (1000, 500, 100), ):
                            config['eval_params'] = sklearn_eval_cfg
                            config['eval_params']['mode'] = mode
                            config['eval_params']['n_train'] = n_train
                            # config['eval_params']['reg_model'] = 'ridge'
                            config['eval_params']['reg_model'] = 'GBTR'
                            config['eval_params']['cls_model'] = 'GBTC'
                            args.tags = ['GBT', ]
                            config['eval_params']['n_fold'] = 1
                            ckpoints = ('last', )
                            # ckpoints = ('epoch=19', 'epoch=39', 'epoch=59',)
                            for ckpoint in ckpoints:
                                config['eval_params']['ckpoint'] = ckpoint
                                scikitlearn_eval(config, args)



if __name__ == '__main__':
    main()