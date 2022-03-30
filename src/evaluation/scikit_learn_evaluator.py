#!/usr/bin/env python
"""
Created by zhenlinx on 02/25/2022
"""

from evaluation import CompGenalizationMetrics

class ScikitLearnEvaluator:
    def __init__(self, backbone, datamodule, mode, n_train=None, reg_model='linear', cls_model='logistic', **kwargs):
        self.backbone = backbone
        self.mode = mode
        self.n_train = n_train
        datamodule.prepare_data()
        datamodule.setup()
        task_types = datamodule.train_dataset.task_types
        self.factor_ids = {'reg': (task_types == 'reg').nonzero()[0],
                           'cls': (task_types == 'cls').nonzero()[0]}
        self.metric_names = {'reg': 'R2', 'cls': 'acc'}
        self.model_names = {'reg': reg_model, 'cls': cls_model}

        self.regression_metric = CompGenalizationMetrics(datamodule.train_dataloader(),
                                                         datamodule.test_dataloader(),
                                                         n_factors=datamodule.train_dataset.n_gen_factors,
                                                         model=reg_model,
                                                         factor_selector=self.factor_ids['reg']
                                                         )
        self.classification_metric = CompGenalizationMetrics(datamodule.train_dataloader(),
                                                             datamodule.test_dataloader(),
                                                             n_factors=datamodule.train_dataset.n_gen_factors,
                                                             model=cls_model,
                                                             factor_selector=self.factor_ids['cls']
                                                             )
        self.factor_names = datamodule.train_dataset.lat_names


    def eval(self):
        results = {'reg': self.regression_metric(self.backbone, mode=self.mode),
                   'cls': self.classification_metric(self.backbone, mode=self.mode)}
        log_dict = {}
        for task, scores in results.items():
            metric_name = self.metric_names[task]
            for i, score in enumerate(scores):
                factor_name = self.factor_names[self.factor_ids[task][i]][:6]
                log_dict[f'{metric_name}_{factor_name}'] = score
            log_dict[f'{metric_name}_mean'] = scores.mean()

        return log_dict

    @property
    def name(self):
        return '{}_{}{}_reg-{}_cls-{}'.format(
            self.__class__.__name__,
            self.mode,
            f'_{self.n_train}train' if self.n_train is not None else '',
            self.model_names['reg'],
            self.model_names['cls'],
        )