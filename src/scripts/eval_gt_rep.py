#!/usr/bin/env python
"""
Evaluate ground truth representation

Created by zhenlinx on 07/31/2022
"""
import os
import sys
sys.path.append(os.path.realpath('..'))
import argparse
import yaml
from itertools import product
import torch
from pytorch_lightning import  seed_everything
from pytorch_lightning.loggers import WandbLogger
from copy import deepcopy

from evaluation import ScikitLearnEvaluator
from datasets import get_datamodule
from scripts.experiments import add_vae_argument, scikitlearn_eval, setup_experiment


def linear_map(x):
    x = x.float()
    latent_size = x.shape[1]
    scale = (torch.rand(latent_size)+1e-6).unsqueeze(0)
    return x*scale

def affine_map(x):
    x = x.float()
    latent_size = x.shape[1]
    scale = (torch.rand(latent_size)+1e-6).unsqueeze(0)
    res = torch.rand(latent_size).unsqueeze(0)
    return x * scale + res

def polynomial_map(x):
    x = x.float()
    latent_size = x.shape[1]
    # scale_2 = (torch.rand(latent_size)+1e-6).unsqueeze(0)
    # scale_1 = (torch.rand(latent_size)+1e-6).unsqueeze(0)
    # scale_0 = (torch.rand(latent_size)+1e-6).unsqueeze(0)
    # # return x ** 2 * scale_2 + x * scale_1 + scale_0
    return x ** 2

rep_mapping_functions = {
    'same': lambda x: x.float(),
    'linear': linear_map,
    'affine': affine_map,
    'polynomial': polynomial_map,
}

class ScikitLearnGTRepEvaluator(ScikitLearnEvaluator):
    def __init__(self, latent_size=None, **kwargs):
        super(ScikitLearnGTRepEvaluator, self).__init__(**kwargs)
        self.mapping = kwargs["mapping"]  # how we map the linear
        self.latent_sizes = self.train_data.dataset.lat_sizes

    def eval(self):
        # get gt rep
        train_X, train_y = self.get_gt_rep(self.train_data)
        test_X, test_y = self.get_gt_rep(self.test_data)

        train_X, test_X = self.make_rep(train_X, test_X)
        train_X, train_y = train_X.numpy(), train_y.numpy()
        test_X, test_y = test_X.numpy(), test_y.numpy()

        results = {'reg': self.regression_metric(model_zs=((train_X, train_y), (test_X, test_y)), mode=self.mode),
                   'cls': self.classification_metric(model_zs=((train_X, train_y), (test_X, test_y)), mode=self.mode)}
        log_dict = {}
        for task, scores in results.items():
            metric_name = self.metric_names[task]
            for i, score in enumerate(scores):
                factor_name = self.factor_names[self.factor_ids[task][i]][:6]
                log_dict[f'{metric_name}_{factor_name}'] = score
            log_dict[f'{metric_name}_mean'] = scores.mean()
        return log_dict

    def get_gt_rep(self, dataloader):
        latents, targets = [], []
        for _, t in dataloader:
            latents.append(deepcopy(t))
            targets.append(deepcopy(t))
        latents = torch.cat(latents)
        targets = torch.cat(targets)
        return latents, targets

    def make_rep(self, reps_train, reps_test):
        reps = torch.cat([reps_train, reps_test])
        reps = (reps + 1) / self.latent_sizes
        n_train, n_test = reps_train.shape[0], reps_test.shape[0]
        f_mapping = rep_mapping_functions[self.mapping]
        reps_mapped = f_mapping(reps)
        reps_train, reps_test = torch.split(reps_mapped, (n_train, n_test))
        return reps_train, reps_test

    @property
    def name(self):
        return '{}_{}{}_reg-{}_cls-{}_{}fold{}'.format(
            self.__class__.__name__,
            self.mapping,
            f'_{self.n_train}train' if self.n_train is not None else '',
            self.model_names['reg'],
            self.model_names['cls'],
            self.n_fold,
            '_rev' if self.reverse_task_type else '',
        )

def eval_gt_rep(config, args):
    seed_everything(config['exp_params']['random_seed'])

    exp_name = 'GTRep_{}'.format(
        config['exp_params']['dataset'],
    )

    exp_root = os.path.join(config['logging_params']['save_dir'],
                            config['exp_params']['dataset'],
                            exp_name, f"version_{config['exp_params']['random_seed']}")

    dm = get_datamodule(config['exp_params']['dataset'],
                        data_dir=config['exp_params']['data_path'],
                        batch_size=config['exp_params']['batch_size'],
                        num_workers=0,
                        n_train=config['eval_params']['n_train'],
                        n_fold=config['eval_params']['n_fold'],
                        random_seed=config['exp_params']['random_seed'],
                        )

    evaluator = ScikitLearnGTRepEvaluator(backbone=None, datamodule=dm, **config['eval_params'])

    print(f"======= {evaluator.name} with {exp_name}  =======")
    ft_root = os.path.join(exp_root, evaluator.name,
                           f"version_{config['exp_params']['random_seed']}")

    if not os.path.isdir(ft_root):
        os.makedirs(ft_root, exist_ok=True)
    eval_res = evaluator.eval()

    ft_logger = None if args.nowb else WandbLogger(project=args.project,
                                                   name=f"{evaluator.name}_{exp_name}",
                                                   save_dir=ft_root,
                                                   tags=['GT_Rep', 'scikit_eval_v2', ] + args.tags,
                                                   config=config,
                                                   reinit=True
                                                   )

    if ft_logger:
        ft_logger.log_hyperparams(config)
        ft_logger.log_metrics(eval_res)
        ft_logger.experiment.finish()
    else:
        print(eval_res)



def main():
    parser = argparse.ArgumentParser(description='Generic runner for VAE models')
    add_vae_argument(parser)
    args = parser.parse_args()
    config, sklearn_eval_cfg = setup_experiment(args)

    # setting hyperparameters
    # for data in ('dsprites90d_random_v5',  'mpi3d_real_random_v5',):
    for data in ('mpi3d_real_random_v5',):
        for mapping in ('polynomial', ):
            for seed in (2001, 2002, 2003):
            # for seed in (2003, ):
                # sklearn eval
                for n_train in (500, ):
                    config['eval_params'] = sklearn_eval_cfg
                    config['eval_params']['n_train'] = n_train
                    config['exp_params'][
                        'data_path'] = 'YOUR_PATH_TO_DATA'
                    config['exp_params']['random_seed'] = seed
                    config['exp_params']['dataset'] = data

                    if 'mpi3d' in data:
                        config['exp_params'][
                            'data_path'] = 'YOUR_PATH_TO_DATA'
                        config['model_params']['input_size'] = [3, 64, 64]

                    if args.gbt:
                        config['eval_params']['reg_model'] = 'GBTR'
                        config['eval_params']['cls_model'] = 'GBTC'
                        args.tags = ['GBT', ]
                    config['eval_params']['mapping'] = mapping
                    eval_gt_rep(config, args)


if __name__ == '__main__':
    main()
