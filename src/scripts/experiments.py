#!/usr/bin/env python
"""
Created by zhenlinx on 03/01/2022
"""
import os
import sys
import shutil
import yaml

import torch
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

from models import vae_models
from datasets import get_datamodule
from evaluation import ScikitLearnEvaluator, DisentangleMetricEvaluator, TopoSimEval
from commons.callbacks import MyModelCheckpoint

sys.path.append(os.path.realpath('..'))


def load_yaml_file(file_name):
    with open(file_name, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)
    return config


def train_vae(config, args):
    seed_everything(config['exp_params']['random_seed'])

    # set experiments
    vae = vae_models[config['model_params']['name']](
        **config['model_params'],
    )

    exp_name = vae.name + '_{}_batch{}_{}epochs'.format(
        config['exp_params']['dataset'],
        config['exp_params']['batch_size'],
        config['exp_params']['max_epochs'],
    )

    exp_root = os.path.join(config['logging_params']['save_dir'],
                            config['exp_params']['dataset'],
                            exp_name, f"version_{config['exp_params']['random_seed']}", 'debug' if args.debug else '')

    ckpoint_path = os.path.join(exp_root, 'checkpoints', 'last.ckpt')

    if os.path.isfile(ckpoint_path) and not args.overwrite and not args.test and not args.compmetric:
        print(f"Exp {exp_name} exsited")
        return

    if not args.finetune and config['exp_params']['max_epochs'] > 0:
        dm = get_datamodule(config['exp_params']['dataset'],
                            data_dir=config['exp_params']['data_path'],
                            batch_size=config['exp_params']['batch_size'],
                            num_workers=config['exp_params']['train_workers'],
                            random_seed=config['exp_params']['random_seed'],
                            virtual_n_samples=config['exp_params']['val_steps'] * config['exp_params']['batch_size'],
                            )

        if not os.path.isdir(exp_root):
            os.makedirs(exp_root, exist_ok=True)

        logger = None if args.nowb else WandbLogger(project=args.project,
                                                    name=exp_name,
                                                    save_dir=exp_root,
                                                    tags=[config['model_params']['name'], 'pretrain'] + args.tags,
                                                    config=config,
                                                    reinit=True
                                                    )
        if logger is not None:
            logger.watch(vae, log="all")
        # Init ModelCheckpoint callback, monitoring 'val_loss'
        best_checkpoint_callback = MyModelCheckpoint(
            dirpath=os.path.join(exp_root, 'checkpoints'),
            monitor='val_loss',
            filename="best",
            every_n_epochs=1,
            verbose=False,
            save_last=False,
        )
        checkpoint_callback = MyModelCheckpoint(
            dirpath=os.path.join(exp_root, 'checkpoints'),
            save_top_k=-1,
            filename="{epoch:02d}",
            every_n_epochs=20,
            verbose=False,
            save_last=True,
        )

        lr_monitor = LearningRateMonitor(logging_interval='epoch')
        callbacks = [checkpoint_callback, best_checkpoint_callback]
        if not args.nowb:
            callbacks.append(lr_monitor)

        # set runner
        runner = Trainer(min_epochs=1,
                         logger=logger,
                         log_every_n_steps=100,
                         num_sanity_val_steps=5,
                         deterministic=True,
                         benchmark=False,
                         max_steps=20 if args.debug else config['exp_params']['train_steps'],
                         val_check_interval=20 if args.debug else config['exp_params']['val_steps'],
                         limit_train_batches=1.0,
                         callbacks=callbacks,
                         # deterministic=True,
                         **config['trainer_params'],
                         )

        if not args.notrain:
            print(f"======= Training {exp_name} =======")
            runner.fit(vae, dm)
            print(f"Best model at {best_checkpoint_callback.best_model_path}")

        best_model_path = best_checkpoint_callback.best_model_path

        if config['exp_params']['max_epochs'] > 0:
            ckpt = torch.load(ckpoint_path, map_location=torch.device('cpu'))
            vae.load_state_dict(ckpt['state_dict'])


        print(f"======= Testing {exp_name} =======")
        if args.test:
            runner.test(vae, datamodule=dm)

        if args.compmetric:
            if 'Recurrent' in config['model_params']['name']:
                evaluator = TopoSimEval(vae, dm)
            else:
                evaluator = DisentangleMetricEvaluator(vae, dm)
            res = evaluator.eval()
            if logger is not None:
                logger.log_metrics(res)
            else:
                print(res)

        if logger is not None:
            logger.experiment.finish()


def scikitlearn_eval(config, args):
    if (not args.test or args.finetune) and (not args.nofinetune):
        # https://github.com/pytorch/pytorch/issues/11201
        # import torch.multiprocessing
        # torch.multiprocessing.set_sharing_strategy('file_system')

        seed_everything(config['exp_params']['random_seed'])

        """learn a task module on learned encoder"""
        vae = vae_models[config['model_params']['name']](
            **config['model_params'],
        )
        if args.gpu is not None:
            vae = vae.to(f'cuda:{args.gpu[0]}')

        exp_name = vae.name + '_{}_batch{}_{}epochs'.format(
            config['exp_params']['dataset'],
            config['exp_params']['batch_size'],
            config['exp_params']['max_epochs'],
        )

        exp_root = os.path.join(config['logging_params']['save_dir'],
                                config['exp_params']['dataset'],
                                exp_name, f"version_{config['exp_params']['random_seed']}")
        ckpoint_path = os.path.join(exp_root, 'checkpoints', f"{config['eval_params']['ckpoint']}.ckpt")

        dm = get_datamodule(config['exp_params']['dataset'],
                            data_dir=config['exp_params']['data_path'],
                            batch_size=config['exp_params']['batch_size'],
                            num_workers=0,
                            n_train=config['eval_params']['n_train'],
                            n_fold=config['eval_params']['n_fold'],
                            random_seed=config['exp_params']['random_seed'],
                            )

        print("Loading checkpoint at {}".format(ckpoint_path))
        ckpt = torch.load(ckpoint_path, map_location=torch.device('cpu'))
        vae.load_state_dict(ckpt['state_dict'])
        print("Checkpoint loaded!")

        evaluator = ScikitLearnEvaluator(
            vae,
            dm,
            **config['eval_params'],
        )
        print(f"======= {evaluator.name} with {exp_name}  =======")

        ft_root = os.path.join(exp_root, evaluator.name,
                               f"version_{config['exp_params']['random_seed']}")

        if not os.path.isdir(ft_root):
            os.makedirs(ft_root, exist_ok=True)
        eval_res = evaluator.eval()

        ft_logger = None if args.nowb else WandbLogger(project=args.project,
                                                       name=f"{evaluator.name}_{exp_name}",
                                                       save_dir=ft_root,
                                                       tags=[config['model_params']['name'], 'scikit_eval_v2', ] + args.tags,
                                                       config=config,
                                                       reinit=True
                                                       )

        if ft_logger:
            ft_logger.log_hyperparams(config)
            ft_logger.log_metrics(eval_res)
            ft_logger.experiment.finish()
        else:
            print(eval_res)


def setup_experiment(args):
    config = load_yaml_file(args.filename)

    sklearn_eval_config_file = 'configs/scikitlearn_eval.yaml'
    sklearn_eval_cfg = load_yaml_file(sklearn_eval_config_file)

    if args.debug:
        config['exp_params']['train_workers'], config['exp_params']['val_workers'] = 0, 0

    config['trainer_params']['gpus'] = args.gpu
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'  # Deterministic behavior of torch.addmm. Please refer to https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility
    print(config['trainer_params']['gpus'])
    return config, sklearn_eval_cfg

def add_vae_argument(parser):
    parser.add_argument('--config', '-c',
                        dest="filename",
                        metavar='FILE',
                        help='path to the config file',
                        default='configs/vae.yaml')
    parser.add_argument('--debug', '-d', action='store_true',
                        help='debug mode')
    parser.add_argument('--test', '-t', action='store_true',
                        help='debug mode')
    parser.add_argument('--notrain', '-ntr', action='store_true',
                        help='do not run test')
    parser.add_argument('--finetune', '-ft', action='store_true',
                        help='finetune mode: only run finetuning on trained vae')
    parser.add_argument('--nofinetune', '-nft', action='store_true',
                        help='do not finetune, only train vae')
    parser.add_argument('--ckpt', '-cp', action='store_true',
                        help='path to checkpoints')
    parser.add_argument('--overwrite', '-ow', action='store_true',
                        help='overwrite existing checkpoint otherwise skip training')
    parser.add_argument('--gpu', '-g', type=int, nargs='+',
                        help='gpu ids')
    parser.add_argument('--version_id', '-v', type=int, default=0,
                        help='running versions')
    parser.add_argument('--hidden_dim', '-hd', type=int,
                        help='hidden dim for finetuner')
    parser.add_argument('--latent_dim', '-ld', type=int,
                        help='latent dim of vae (length of discrete codes)')
    parser.add_argument('--beta', '-bt', type=float, default=1.0,
                        help='coefficiency of kl_loss term')
    parser.add_argument('--tags', '-tg', type=str, nargs='+', default=[],
                        help='tags add to experiments')
    parser.add_argument('--project', '-pj', type=str, default='discrete_comp_gen_thesis',
                        help='the name of project for W&B logger ')
    parser.add_argument('--multi', '-mt', action='store_true',
                        help='run multiDsprits exp')
    parser.add_argument('--nowb', '-nw', action='store_true',
                        help='do not run log on weight and bias')
    parser.add_argument('--sklearn', '-sk', action='store_true',
                        help='use scikit-learn evaluator')
    parser.add_argument('--compmetric', '-cm', action='store_true',
                            help='test the disentanglement score with dis-lib metrics')