#!/usr/bin/env python
"""
Created by zhenlinx on 02/25/2022
"""

from evaluation import CompGenalizationMetrics
from .utils import infer

class ScikitLearnEvaluator:
    def __init__(self, backbone, datamodule, mode,
                 n_train=None, n_fold=1, reg_model='linear', cls_model='logistic', reverse_task_type=False,
                 ckpoint='', **kwargs):
        """

        :param backbone:
        :param datamodule:
        :param mode:
        :param n_train: number of labeled samples for training a supervised model
        :param n_fold: each fold has different n_train samples and results in a model,
            average performance of n_fold models are reported.
        :param reg_model:
        :param cls_model:
        :param reverse_task_type: add metrics that use reg models for cls factors and cls models for reg factors
        :param kwargs:
        """
        self.backbone = backbone
        self.mode = mode
        self.n_train = n_train
        self.n_fold = n_fold
        self.reverse_task_type = reverse_task_type
        datamodule.prepare_data()
        datamodule.setup()
        task_types = datamodule.train_dataset.task_types
        if not reverse_task_type:
            self.factor_ids = {'reg': (task_types == 'reg').nonzero()[0],
                               'cls': (task_types == 'cls').nonzero()[0]}
        else:
            self.factor_ids = {'reg': (task_types == 'cls').nonzero()[0],
                               'cls': (task_types == 'reg').nonzero()[0]}
        self.metric_names = {'reg': 'R2', 'cls': 'acc'}
        self.model_names = {'reg': reg_model, 'cls': cls_model}

        self.regression_metric = CompGenalizationMetrics(train_data=None,
                                                         test_data=None,
                                                         n_factors=datamodule.train_dataset.num_factors,
                                                         model=reg_model,
                                                         factor_selector=self.factor_ids['reg'],
                                                         n_fold = n_fold
                                                         )
        self.classification_metric = CompGenalizationMetrics(train_data=None,
                                                             test_data=None,
                                                             n_factors=datamodule.train_dataset.num_factors,
                                                             model=cls_model,
                                                             factor_selector=self.factor_ids['cls'],
                                                             n_fold=n_fold
                                                             )

        self.train_data = datamodule.train_dataloader()
        self.test_data = datamodule.test_dataloader()

        self.factor_names = datamodule.train_dataset.lat_names
        self.checkpoint_name = ckpoint

    def eval(self):
        train_X, train_y = infer(self.backbone, self.train_data, self.mode)
        train_X, train_y = train_X.numpy(), train_y.numpy()
        test_X, test_y = infer(self.backbone, self.test_data, self.mode)
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

    @property
    def name(self):
        return '{}_{}{}_reg-{}_cls-{}_{}fold{}_ckpt{}'.format(
            self.__class__.__name__,
            self.mode,
            f'_{self.n_train}train' if self.n_train is not None else '',
            self.model_names['reg'],
            self.model_names['cls'],
            self.n_fold,
            '_rev' if self.reverse_task_type else '',
            self.checkpoint_name,
        )