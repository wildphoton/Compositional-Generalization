#!/usr/bin/env python

from torchmetrics import Metric
import torch.nn as nn
import torch

class GroupMetric(nn.Module):
    """
    A group metric to handle multiple same-type metrics e.g. multiple Accuracy
    """

    def __init__(self, metric_class, group_size, names=None, **kwargs):
        super(GroupMetric, self).__init__()
        self.names = names
        if self.names is not None:
            assert len(names) == group_size
        self.group_metrics = nn.ModuleList([metric_class(**kwargs) for i in range(group_size)])

    def update(self, preds, targets):
        for i, metric in enumerate(self.group_metrics):
            metric.update(preds[i], targets[i])

    def compute(self):
        if self.names is None:
            return [metric.compute() for i, metric in enumerate(self.group_metrics)]
        else:
            return {self.names[i]: metric.compute() for i, metric in enumerate(self.group_metrics)}

    def forward(self, preds, targets, **kwargs):
        """
        preds and targets are iterators over pairs of predictions and targets
        :param preds:
        :param targets:
        :param kwargs:
        :return:
        """
        if self.names is None:
            return [metric(preds[i], targets[i], **kwargs) for i, metric in enumerate(self.group_metrics)]
        else:
            return {self.names[i]: metric(preds[i], targets[i], **kwargs) for i, metric in enumerate(self.group_metrics)}

    def mean(self):
        if self.names is None:
            return torch.mean(torch.FloatTensor(self.compute()))
        else:
            return torch.mean(torch.FloatTensor(list(self.compute().values())))

    def reset(self):
        for metric in self.group_metrics:
            metric.reset()