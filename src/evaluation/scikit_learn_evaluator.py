#!/usr/bin/env python
"""
Created by zhenlinx on 02/25/2022
"""

from evaluation import CompGenalizationMetrics

class ScikitLearnEvaluator:
    def __init__(self, backbone, datamodule, mode, n_train=None, **kwargs):
        self.backbone = backbone
        self.mode = mode
        self.n_train = n_train
        datamodule.prepare_data()
        datamodule.setup()
        self.regression_metric = CompGenalizationMetrics(datamodule.train_dataloader(),
                                         datamodule.test_dataloader(),
                                         n_factors=datamodule.train_dataset.n_gen_factors,
                                         regressor='ridge',
                                         )
        self.classification_metric = CompGenalizationMetrics(datamodule.train_dataloader(),
                                             datamodule.test_dataloader(),
                                             n_factors=datamodule.train_dataset.n_gen_factors,
                                             regressor='logistic',
                                             )
        self.factor_names = datamodule.train_dataset.lat_names

    def eval(self):
        results = {'R2': self.regression_metric(self.backbone, mode=self.mode),
                   'acc': self.classification_metric(self.backbone, mode=self.mode)}
        log_dict = {}
        for metric, scores in results.items():
            for i, score in enumerate(scores):
                name = self.factor_names[i][:6]
                log_dict[f'{metric}_{name}'] = score
            log_dict[f'{metric}_mean'] = scores.mean()

        return log_dict

    @property
    def name(self):
        return '{}_{}{}_reg{}_cls{}'.format(
            self.__class__.__name__,
            self.mode,
            f'_{self.n_train}train' if self.n_train is not None else '',
            'Ridge',
            'Logistic',
        )