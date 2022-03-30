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
from scripts.experiments import add_vae_argument, train_vae, scikitlearn_eval, finetune_eval, setup_experiment

def main():
    parser = argparse.ArgumentParser(description='Generic runner for VAE models')
    add_vae_argument(parser)
    args = parser.parse_args()
    args.filename = 'configs/discrete_vae.yaml'
    config, sklearn_eval_cfg, linear_eval_cfg = setup_experiment(args)

    versions = ((3,),)
    # for data in ['dsprites90d']:
    for data in ['mpi3d_real']:
        for gen_type in ['random', ]:
            for version in versions[args.version_id]:
                for seed in (2001, ):
                    for recon_loss, beta, latent_size, dict_size, arch in product(('bce', ),
                                                                                  (0.1, ), (10, ), (256, ),
                                                                                  # ('burgess', 'burgess_wide')
                                                                                  ('burgess_wide4x', )
                                                                                  ):

                        config['model_params']['name'] = 'RecurrentDiscreteVAE'
                        config['model_params']['beta'] = beta
                        config['model_params']['latent_size'] = latent_size
                        config['model_params']['dictionary_size'] = dict_size
                        config['model_params']['recon_loss'] = recon_loss
                        config['model_params']['fix_length'] = False
                        config['model_params']['architecture'] = arch

                        config['exp_params']['random_seed'] = seed
                        config['exp_params']['max_epochs'] = 200
                        # config['exp_params']['batch_size'] = 512

                        config['exp_params']['dataset'] = '{}_{}_v{}'.format(data, gen_type, version)
                        if 'mpi3d' in data:
                            config['exp_params'][
                                'data_path'] = '/playpen-raid2/zhenlinx/Data/disentanglement/mpi3d'
                            # config['exp_params']['max_epochs'] = 200  # 100 for dsprites and 50 for mpi3d
                            config['model_params']['input_size'] = [3, 64, 64]

                        train_vae(config, args)

                        if args.sklearn:
                            # sklearn eval
                            # for mode, n_train in product(('post', ), (10000, ), ):
                            for mode, n_train in product(('post',  'pre', ), (1000, 100), ):
                                config['eval_params'] = sklearn_eval_cfg
                                config['eval_params']['mode'] = mode
                                config['eval_params']['n_train'] = n_train
                                config['eval_params']['ckpoint'] = 'last'
                                config['eval_params']['reg_model'] = 'ridge'
                                ckpoints = ('last',  'epoch=49') if 'mpi3d' in data else ('last',)
                                for ckpoint in ckpoints:
                                    config['eval_params']['ckpoint'] = ckpoint
                                    scikitlearn_eval(config, args)
                        else:
                            # eval with training a linear classifier or mlp
                            for hidden_dim, mode, n_train, sampling in product((None, ), ('latent', 'pre', 'post'), (1000, ), (False, )):
                                config['eval_params'] = linear_eval_cfg
                                config['eval_params']['mode'] = mode
                                config['eval_params']['n_train'] = n_train
                                config['eval_params']['hidden_dim'] = hidden_dim

                                config['eval_params']['learning_rate'] = 0.001
                                config['eval_params']['optim'] = 'Adam'
                                config['eval_params']['scheduler_type'] = 'warmup_cosine'
                                config['eval_params']['warmup_epochs'] = 10
                                config['eval_params']['ckpoint'] = 'last'

                                config['eval_params']['train_steps'] = 200000
                                config['eval_params']['val_steps'] = 2000

                                config['eval_params']['sampling'] = sampling
                                finetune_eval(config, args)



if __name__ == '__main__':
    main()
