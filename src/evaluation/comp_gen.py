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
    def __init__(self, train_data, test_data, n_factors,
                 regressor='lasso',
                 regressoR_coeffkwargs=None,
                 **kwargs):
        kwargs.update({'cv': 5,})

        if regressoR_coeffkwargs is not None:
            kwargs.update(regressoR_coeffkwargs)

        if regressor == 'lasso':
            if 'alphas' not in kwargs:
                kwargs['alphas'] = [0.00005, 0.0001, 0.001,]
            if 'selection' not in kwargs:
                kwargs['selection'] ='random'
            regressor = LassoCV(**kwargs)
        elif regressor == 'ridge':
            if 'alphas' not in kwargs:
                kwargs['alphas'] = [0.01, 0.1, 1.0, 10]
            # if 'normalize' not in kwargs:
            #     kwargs['normalize'] = True
            regressor = RidgeCV(**kwargs)
        elif regressor == 'logistic':
            regressor = LogisticRegressionCV(**kwargs)
        elif regressor == 'linear':
            if 'cv' in kwargs:
                del kwargs['cv']
            regressor = LinearRegression(**kwargs)
        elif regressor == 'random-forest':
            regressor = RandomForestRegressor(**kwargs)
        else:
            raise ValueError()

        self.train_data = train_data
        self.test_data = test_data
        self.n_factors = n_factors
        self.regressor = regressor

    def compute_score(self, model, model_zs=None, mode='latent'):
        if model_zs is None:
            train_X, train_y = infer(model, self.train_data, mode)
            train_X, train_y = train_X.numpy(), train_y.numpy()
            test_X, test_y = infer(model, self.test_data, mode)
            test_X, test_y = test_X.numpy(), test_y.numpy()
        else:
            (train_X, train_y), (test_X, test_y) = model_zs

        # train_X = (train_X - train_X.mean(axis=0))   / (train_X.std(axis=0) + EPS)
        # test_X = (test_X - test_X.mean(axis=0))   / (test_X.std(axis=0) + EPS)
        # train_y = (train_y - train_y.mean(axis=0)) / (train_y.std(axis=0) + EPS)
        # test_y = (test_y - test_y.mean(axis=0)) / (test_y.std(axis=0) + EPS)
        if type(self.regressor) == LogisticRegressionCV:
            train_y = train_y.astype(int)
            test_y = test_y.astype(int)

        R2 = []
        for k in range(self.n_factors):
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