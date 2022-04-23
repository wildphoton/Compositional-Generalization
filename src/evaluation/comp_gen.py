#!/usr/bin/env python
"""
Created by zhenlinx on 02/25/2022
"""
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
from dataclasses import dataclass
from torch.utils.data import DataLoader
from sklearn.linear_model import LassoCV, MultiTaskLassoCV, RidgeCV, LogisticRegressionCV, LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from .utils import infer

EPS = 1e-12

class CompGenalizationMetrics:
    def __init__(self, train_data, test_data,
                 n_factors,
                 n_fold=1,
                 factor_selector = None,
                 model='lasso',
                 regressoR_coeffkwargs=None,
                 **kwargs):
        kwargs.update({'cv': 5,})

        if regressoR_coeffkwargs is not None:
            kwargs.update(regressoR_coeffkwargs)

        if model == 'lasso':
            if 'alphas' not in kwargs:
                kwargs['alphas'] = [0.00005, 0.0001, 0.001,]
            if 'selection' not in kwargs:
                kwargs['selection'] ='random'
            self.model_class = LassoCV
        elif model == 'ridge':
            if 'alphas' not in kwargs:
                kwargs['alphas'] = [0, 0.01, 0.1, 1.0, 10]
            # if 'normalize' not in kwargs:
            #     kwargs['normalize'] = True
            self.model_class = RidgeCV
        elif model == 'logistic':
            self.model_class = LogisticRegressionCV
        elif model == 'linear':
            if 'cv' in kwargs:
                del kwargs['cv']
            self.model_class = LinearRegression
        elif model == 'random-forest':
            self.model_class = RandomForestRegressor
        else:
            raise ValueError()

        self.train_data = train_data
        self.test_data = test_data
        self.n_fold = n_fold
        self.factor_indices = factor_selector if factor_selector is not None else list(range(n_factors))
        self.kwargs = kwargs
        self.reset_model()

    def reset_model(self):
        self.model = self.model_class(**self.kwargs)

    def compute_score(self, rep_model, model_zs=None, mode='latent'):
        if model_zs is None:
            train_X, train_y = infer(rep_model, self.train_data, mode)
            train_X, train_y = train_X.numpy(), train_y.numpy()
            test_X, test_y = infer(rep_model, self.test_data, mode)
            test_X, test_y = test_X.numpy(), test_y.numpy()
        else:
            (train_X, train_y), (test_X, test_y) = model_zs

        if self.model_class == LogisticRegressionCV:
            train_y = train_y.astype(int)
            test_y = test_y.astype(int)

        score = []
        n_samples_per_fold = train_X.shape[0] // self.n_fold
        for k in self.factor_indices:
            train_y_k = train_y[:, k]
            score_k = []
            for j in range(self.n_fold):
                train_X_fold = train_X[n_samples_per_fold*j:n_samples_per_fold*(j+1)]
                train_y_k_one_fold = train_y_k[n_samples_per_fold*j:n_samples_per_fold*(j+1)]
                if len(np.unique(train_y_k)) > 1:
                    try:
                        self.reset_model()
                        self.model.fit(train_X_fold, train_y_k_one_fold)
                        score_k.append(self.model.score(test_X, test_y[:, k]))
                    except Exception as e:
                        print("Error message {}".format(str(e)))
            score.append(np.nanmean(score_k))
        return np.array(score)

    def __call__(self, model=None, model_zs=None, mode='latent'):
        return self.compute_score(model, model_zs, mode)

def compute_compgen_metric(models, train_data, test_data):
    """
    Convenience function to compute the DCI metrics for a set of models
    in a given dataset.
    """
    n_factors = train_data.num_factors

    train_loader = DataLoader(train_data, batch_size=64, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_data, batch_size=64, num_workers=4, pin_memory=True)

    com_gen = CompGenalizationMetrics(train_loader, test_loader, n_factors=n_factors)

    results = [com_gen(model) for model in models]

    return results