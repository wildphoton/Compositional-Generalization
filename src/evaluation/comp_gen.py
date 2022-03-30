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
            model = LassoCV(**kwargs)
        elif model == 'ridge':
            if 'alphas' not in kwargs:
                kwargs['alphas'] = [0, 0.01, 0.1, 1.0, 10]
            # if 'normalize' not in kwargs:
            #     kwargs['normalize'] = True
            model = RidgeCV(**kwargs)
        elif model == 'logistic':
            model = LogisticRegressionCV(**kwargs)
        elif model == 'linear':
            if 'cv' in kwargs:
                del kwargs['cv']
            model = LinearRegression(**kwargs)
        elif model == 'random-forest':
            model = RandomForestRegressor(**kwargs)
        else:
            raise ValueError()

        self.train_data = train_data
        self.test_data = test_data
        self.factor_indices = factor_selector if factor_selector is not None else list(range(n_factors))
        self.regressor = model

    def compute_score(self, model, model_zs=None, mode='latent'):
        if model_zs is None:
            train_X, train_y = infer(model, self.train_data, mode)
            train_X, train_y = train_X.numpy(), train_y.numpy()
            test_X, test_y = infer(model, self.test_data, mode)
            test_X, test_y = test_X.numpy(), test_y.numpy()
        else:
            (train_X, train_y), (test_X, test_y) = model_zs

        if type(self.regressor) == LogisticRegressionCV:
            train_y = train_y.astype(int)
            test_y = test_y.astype(int)

        R2 = []
        for k in self.factor_indices:
            train_y_k = train_y[:, k]
            if len(np.unique(train_y_k)) > 1:
                self.regressor.fit(train_X, train_y_k)
                R2.append(self.regressor.score(test_X, test_y[:, k]))

        return np.array(R2)

    def __call__(self, model, model_zs=None, mode='latent'):
        return self.compute_score(model, model_zs, mode)

def compute_compgen_metric(models, train_data, test_data):
    """
    Convenience function to compute the DCI metrics for a set of models
    in a given dataset.
    """
    n_factors = train_data.n_gen_factors

    train_loader = DataLoader(train_data, batch_size=64, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_data, batch_size=64, num_workers=4, pin_memory=True)

    com_gen = CompGenalizationMetrics(train_loader, test_loader, n_factors=n_factors)

    results = [com_gen(model) for model in models]

    return results