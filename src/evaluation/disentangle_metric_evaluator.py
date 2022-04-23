#!/usr/bin/env python
"""
Created by zhenlinx on 04/15/2022
"""

import sys
import os
import gin
import gin.tf
import numpy as np
import torch

gin.enter_interactive_mode()

# needed later:
from disentanglement_lib.evaluation.metrics import beta_vae
from disentanglement_lib.evaluation.metrics import dci
from disentanglement_lib.evaluation.metrics import downstream_task
from disentanglement_lib.evaluation.metrics import factor_vae
from disentanglement_lib.evaluation.metrics import fairness
from disentanglement_lib.evaluation.metrics import irs
from disentanglement_lib.evaluation.metrics import mig
from disentanglement_lib.evaluation.metrics import modularity_explicitness
from disentanglement_lib.evaluation.metrics import reduced_downstream_task
from disentanglement_lib.evaluation.metrics import sap_score
from disentanglement_lib.evaluation.metrics import unsupervised_metrics

# IRS, DCI, Factor-VAE, MIG, SAP-Score
config_root = '../evaluation/extra_metrics_configs'
metric_score_name = {
                'dci': 'disentanglement',
                'mig': 'discrete_mig',
                'sap_score': 'SAP_score',
                'irs': 'IRS',
                }

class DisentangleMetricEvaluator():
    def __init__(self, model, datamodule, metrics=('dci', 'sap_score', 'mig', 'irs')):
        self.model = model
        datamodule.prepare_data()
        datamodule.setup()
        self.train_dataset = datamodule.train_dataset
        self.test_dataset = datamodule.test_dataset
        self.metrics = metrics

    def eval(self):
        res_train = eval_disentangle_metrics(self.model, self.train_dataset, self.metrics)
        res_test = eval_disentangle_metrics(self.model, self.test_dataset, self.metrics)
        res = {}
        for metric_name, score in res_train.items():
            res[f'train_{metric_name}'] = score
        for metric_name, score in res_test.items():
            res[f'test_{metric_name}'] = score
        return res

def eval_disentangle_metrics(model, dataset, metrics):
    device = next(model.parameters()).device
    random_seed = np.random.randint(2**16)
    def representation_function(x):
        if x.shape[-1] == 3 or x.shape[-1] == 1:
            x = np.transpose(x, (0, 3, 1, 2))
        representation = model.embed(torch.from_numpy(x).float().to(device), mode='latent')
        return np.array(representation.detach().cpu())

    @gin.configurable("evaluation")
    def evaluate_model(evaluation_fn=gin.REQUIRED, random_seed=gin.REQUIRED):
        return evaluation_fn(
            dataset,
            representation_function,
            random_state=np.random.RandomState(random_seed))

    results = {}
    for metric in metrics:
        metric_config = f"{metric}.gin"
        eval_bindings = [
            f'evaluation.random_seed = {random_seed}']
        gin.parse_config_files_and_bindings(
            [os.path.join(config_root, metric_config)], eval_bindings)
        # print(f'Eval metric {metric}')
        with torch.no_grad():
            out = evaluate_model()
        gin.clear_config()
        results[metric] = out[metric_score_name[metric]]
        print(f'Metric {metric}: {results[metric]}')
    return results

# def evaluate_dislib_metrics(model,
#                             dataset,
#                             split_name: str,
#                             dis_lib_metrics: List[str] = ['dci', 'factor_vae_metric', 'sap_score', 'mig', 'mcc', 'irs'],
#                             supervised=False):
#
#     metrics = [f'{m}.gin' for m in dis_lib_metrics]
#     with torch.no_grad():
#         out = evaluate(model, dataset, metrics, supervised=supervised)
#     metric_names = {'mcc': 'meanabscorr',
#                     'dci': 'disentanglement',
#                     'mig': 'discrete_mig',
#                     'sap_score': 'SAP_score',
#                     'modularity_explicitness': 'modularity_score',
#                     'irs': 'IRS',
#                     }
#     for metric, infos in out.items():
#         print(f'dis_lib_{split_name}/{metric}', infos[metric_names[metric]])
#
# def evaluate(model, dataset, eval_config_files=None, device='cuda'):
#     # device = 'cuda' if torch.cuda.is_available() else 'cpu'
#     model = model.to(device)
#
#     def model_wrapper(x):
#         if x.shape[-1] == 3 or x.shape[-1] == 1:
#             x = np.transpose(x, (0, 3, 1, 2))
#         representation = model.embed(torch.from_numpy(x).float().to(device), mode='latent')
#         return np.array(representation.detach().cpu())
#
#     @gin.configurable("evaluation")
#     def evaluate_model(
#             evaluation_fn=gin.REQUIRED,
#             random_seed=gin.REQUIRED):
#         experiment_timer = time.time()
#         results_dict = evaluation_fn(
#             dataset,
#             model_wrapper,
#             random_state=np.random.RandomState(random_seed))
#         results_dict["elapsed_time"] = time.time() - experiment_timer
#         return results_dict
#
#     random_state = np.random.RandomState(0)
#     if eval_config_files is None:
#         eval_config_files = [f for f in os.listdir(config_dir) if
#                              not (f.startswith('.') or 'others' in f)]
#
#     all_results = {}
#     print(eval_config_files)
#     for eval_config in eval_config_files:
#         t0 = time.time()
#
#         metric_name = os.path.basename(eval_config).replace(".gin", "")
#
#         eval_bindings = [
#             f'evaluation.random_seed = {random_state.randint(2 ** 32)}']
#         gin.parse_config_files_and_bindings(
#             [os.path.join(config_dir, eval_config)], eval_bindings)
#         print(f'Eval metric {metric_name}')
#         out = evaluate_model()
#         gin.clear_config()
#         print(f'metric {metric_name} took', time.time() - t0, 's')
#         all_results[metric_name] = out
#     return all_results