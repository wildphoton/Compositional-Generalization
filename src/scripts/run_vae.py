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
    config, sklearn_eval_cfg, linear_eval_cfg = setup_experiment(args)

    # setting hyperparameters
    versions = ((1,),)
    for data in ['dsprites90d']:
        for gen_type in ['random', ]:
            for version in versions[args.version_id]:
                for seed in (2001, ):
                    for recon_loss, beta in product(('mse', ), (0.1, )):

                        config['model_params']['beta'] = beta
                        config['model_params']['latent_size'] = 10
                        config['model_params']['recon_loss'] = recon_loss

                        config['exp_params']['random_seed'] = seed
                        config['exp_params']['max_epochs'] = 100

                        config['exp_params']['dataset'] = '{}_{}_v{}'.format(data, gen_type, version)

                        train_vae(config, args)

                        if args.sklearn:
                            # sklearn eval
                            # for mode, n_train in product(('post', ), (1000, ), ):
                            for mode, n_train in product(('latent', 'pre', 'post'), (1000, ), ):
                                config['eval_params'] = sklearn_eval_cfg
                                config['eval_params']['mode'] = mode
                                config['eval_params']['n_train'] = n_train
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

                                config['eval_params']['train_steps'] = 200000
                                config['eval_params']['val_steps'] = 2000

                                config['eval_params']['sampling'] = sampling
                                finetune_eval(config, args)



if __name__ == '__main__':
    main()
