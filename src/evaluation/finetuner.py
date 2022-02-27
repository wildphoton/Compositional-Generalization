#!/usr/bin/env python
"""
Created by zhenlinx on 03/31/2021
"""
from itertools import chain
from typing import *
import torch
import torch.nn.functional as F
import torch.nn as nn
import pytorch_lightning as pl
import pl_bolts
from torchmetrics import Accuracy, MeanSquaredError, R2Score
from datasets import DSprites
from utils.group_metric import GroupMetric


class TaskHeadMLP(torch.nn.Module):
    """
    A representation evaluatior that use a linear layer or MLP for a or multiple classification tasks
    """
    def __init__(self, n_input, n_classes, n_hidden=None):
        super(TaskHeadMLP, self).__init__()
        if type(n_classes) is int:
            n_classes = (n_classes)
        self.n_input = n_input
        self.n_classes = n_classes
        self.n_hidden = n_hidden

        if n_hidden is not None:
            # if n_hidden is defined, use a simple MLP classifier otherwise a linear classifier
            self.hidden_layer = nn.Sequential(
                # nn.Dropout(p=p),
                nn.Linear(n_input, n_hidden, bias=True),
                nn.BatchNorm1d(n_hidden),
                nn.ReLU(inplace=True),
            )
        output_layers = []
        for output_dim in n_classes:
            output_layers.append(nn.Linear(n_input if self.n_hidden is None else n_hidden, output_dim))
        self.output_layers = nn.ModuleList(output_layers)


    def forward(self, x):
        if self.n_hidden is not None:
            x = self.hidden_layer(x)
        outputs = [cls(x) for cls in self.output_layers]
        return outputs


class RepFineTuner(pl.LightningModule):
    """
    Finetune on top of an learned representation model with the linear protocal or a MLP.
    """

    def __init__(self,
                 backbone,
                 dataset,
                 num_classes: int or list,
                 factor_names: list = None,
                 hidden_dim: Optional[int] = None,
                 # epochs: int = 100,
                 train_steps: int = 1000000,
                 val_steps: int = 10000,
                 learning_rate: float = 0.1,
                 weight_decay: float = 1e-6,
                 nesterov: bool = False,
                 scheduler_type: str = 'cosine',
                 decay_epochs: List = [60, 80],
                 warmup_epochs: int = 0,
                 gamma: float = 0.1,
                 final_lr: float = 0.,
                 n_train: int = None,
                 mode: str = 'latent',
                 finetune_backbone: bool = False,
                 sampling: bool = False,
                 optim: str = '',
                 **kwargs
                 ):
        """

        :param backbone:
        :param in_features:
        :param num_classes:
        :param epochs:
        :param hidden_dim:
        :param learning_rate:
        :param weight_decay:
        :param nesterov:
        :param scheduler_type:
        :param decay_epochs:
        :param gamma:
        :param final_lr:
        """
        super(RepFineTuner, self).__init__()
        self.save_hyperparameters()

        self.learning_rate = learning_rate
        self.nesterov = nesterov
        self.weight_decay = weight_decay

        self.scheduler_type = scheduler_type
        self.decay_epochs = decay_epochs
        self.gamma = gamma
        # self.epochs = epochs
        self.final_lr = final_lr
        self.warmup_epochs = warmup_epochs
        self.train_steps = train_steps
        self.val_steps = val_steps
        self.optim = optim
        self.kwargs = kwargs

        self.backbone = backbone
        self.mode = mode
        self.sampling = sampling
        self.finetune_backbone = finetune_backbone
        self.in_features = self.backbone.get_rep_size(self.mode)
        self.dataset = dataset
        self.hidden_dim = hidden_dim
        self.n_classes = num_classes
        self.task_head = TaskHeadMLP(n_input=self.in_features, n_classes=num_classes, n_hidden=hidden_dim)
        if type(num_classes) is int:
            num_classes = (num_classes)
        self.n_train = n_train
        self.automatic_optimization = True

        self.factor_names = factor_names
        try:
            assert len(self.factor_names) == len(self.n_classes)
            self.factor_names = [name[:6] for name in self.factor_names]
        except:
            self.factor_names = [str(i) for i in range(len(self.n_classes))]

        # setup metrics
        self.train_acc = GroupMetric(Accuracy, len(self.n_classes), names=[f'trAcc_{name}' for name in self.factor_names])
        self.val_acc = GroupMetric(Accuracy, len(self.n_classes), names=[f'vlAcc_{name}' for name in self.factor_names])
        self.test_acc = GroupMetric(Accuracy, len(self.n_classes), names=[f'Acc_{name}' for name in self.factor_names])

        if self.dataset.split('_')[0] in ['dsprites', 'dsprites90d']:
            for class_name in DSprites.lat_names:
                self.register_buffer(f'latent_val_{class_name}',
                                     torch.from_numpy(DSprites.latents_values[class_name]).float())
            self.train_error = GroupMetric(R2Score, len(self.n_classes), names=[f'trR2_{name}' for name in self.factor_names])
            self.val_error = GroupMetric(R2Score, len(self.n_classes), names=[f'vlR2_{name}' for name in self.factor_names])
            self.test_error = GroupMetric(R2Score, len(self.n_classes), names=[f'R2_{name}' for name in self.factor_names])

    def on_train_epoch_start(self) -> None:
        if not self.finetune_backbone:
            self.backbone.eval()

    def shared_step(self, batch):
        # if 'dsprites' in self.dataset:
        #     x, y_all = batch
        #     y = y_all[0]
        # else:
        x, y = batch

        if self.finetune_backbone:
            feats = self.backbone.embed(x, self.mode, self.sampling)
        else:
            with torch.no_grad():
                feats = self.backbone.embed(x, self.mode, self.sampling)

        feats = feats.reshape(feats.size(0), -1)

        logits_list = self.task_head(feats)
        loss_list = [F.cross_entropy(logits_list[i], y[:, i]) for i, loss in enumerate(logits_list)]
        loss = torch.stack(loss_list, dim=0).mean()

        preds_list = [torch.max(logit.detach(), dim=1)[1] for logit in logits_list]
        y_list = [y[:, i] for i in range(y.size(1))]
        return loss, loss_list, logits_list, preds_list, y_list

    def get_lat_val(self, pred_cls):
        # map class prediction into actual values and get MSE
        if 'dsprites' in self.dataset.split('_')[0]:
            return [torch.gather(getattr(self, f'latent_val_{class_name}'), 0, pred_cls[i])
                    for i, class_name in enumerate(DSprites.lat_names)]
        else:
            raise ValueError(f'only dsprites dataset has latent values but get {self.dataset}')

    def update_metrics(self, preds, targets, metric_func, metric_name):
        # return {f'{metric_name}_{self.class_names[i]}': metric_val for i, metric_val in enumerate(metric_func(preds, targets))}
        return metric_func(preds, targets)

    def get_metrics(self, metric_func, metric_name):
        """this will get the accumulated metric"""
        # return {f'{metric_name}_{self.class_names[i]}': metric_val for i, metric_val in enumerate(metric_func.compute())}
        return metric_func.compute()

    def training_step(self, batch, batch_idx):
        loss, loss_list, logits_list, preds_list, y_list = self.shared_step(batch)
        loss = sum(loss_list) / len(loss_list)
        loss_dict = {f'train_loss_{i}': loss for i, loss in enumerate(loss_list)}
        self.log_dict(loss_dict, prog_bar=False)
        self.log('train_loss', loss, prog_bar=False)

        acc_dict = self.update_metrics(preds_list, y_list, self.train_acc, 'train_acc')
        self.log_dict(acc_dict, prog_bar=False)
        self.log('trAcc_mean', self.train_acc.mean(), prog_bar=True)

        if self.dataset.split('_')[0] in ['dsprites', 'dsprites90d']:
            r2_dict = self.update_metrics(self.get_lat_val(preds_list), self.get_lat_val(y_list),
                                           self.train_error, 'train_mse')
            self.log_dict(r2_dict, prog_bar=False)
            self.log('trR2_mean', self.train_error.mean(), prog_bar=True)

        if not self.automatic_optimization:
            opt = self.optimizers()
            opt.zero_grad()
            self.manual_backward(loss)
            opt.step()
        return loss

    def validation_step(self, batch, batch_idx):
        loss, loss_list, logits_list, preds_list, y_list = self.shared_step(batch)
        loss_dict = {f'val_loss_{i}': loss for i, loss in enumerate(loss_list)}
        self.log_dict(loss_dict, prog_bar=False, on_epoch=True)
        self.log('val_loss', loss, prog_bar=False, on_epoch=True)

        self.update_metrics(preds_list, y_list, self.val_acc, 'val_acc')
        # self.log_dict(acc_dict, prog_bar=True, on_epoch=False, on_step=False)
        if self.dataset.split('_')[0] in ['dsprites', 'dsprites90d']:
            self.update_metrics(self.get_lat_val(preds_list), self.get_lat_val(y_list),
                                           self.val_error, 'val_mse')
        return loss

    def test_step(self, batch, batch_idx):
        loss, loss_list, logits_list, preds_list, y_list = self.shared_step(batch)

        self.update_metrics(preds_list, y_list, self.test_acc, 'test_acc')
        if self.dataset.split('_')[0] in ['dsprites', 'dsprites90d']:
            self.update_metrics(self.get_lat_val(preds_list), self.get_lat_val(y_list),
                                           self.test_error, 'test_mse')
        self.log('test_loss', loss, prog_bar=False, on_epoch=True)

    def training_epoch_end(self, outputs: List[Any]) -> None:
        # acc_dict = self.get_metrics(self.train_acc, 'train_acc')
        # self.log_dict(acc_dict, prog_bar=True)
        self.train_acc.reset()
        self.train_error.reset()
        pass

    def validation_epoch_end(self, outputs: List[Any]) -> None:
        acc_dict = self.get_metrics(self.val_acc, 'val_acc')
        self.log_dict(acc_dict, prog_bar=False)
        self.log('vlAcc_mean', self.val_acc.mean(), prog_bar=True)
        if self.dataset.split('_')[0] in ['dsprites', 'dsprites90d']:
            mse_dict = self.get_metrics(self.val_error, 'val_mse')
            self.log_dict(mse_dict, prog_bar=False)
            self.log('vlR2_mean', self.val_error.mean(), prog_bar=True)

        if not self.automatic_optimization:
            sch = self.lr_schedulers()
            try:
                sch.step()
            except:
                sch.step(torch.tensor(outputs).mean().detach())
        self.val_acc.reset()
        self.val_error.reset()

    def test_epoch_end(self, outputs: List[Any]) -> None:
        metric_dict = self.get_metrics(self.test_acc, 'test_acc')
        metric_dict['acc_mean'] = self.test_acc.mean()
        if self.dataset.split('_')[0] in ['dsprites', 'dsprites90d']:
            mse_dict = self.get_metrics(self.test_error, 'test_mse')
            self.log_dict(mse_dict, prog_bar=False)
            metric_dict.update(mse_dict)
            metric_dict['R2_mean'] = self.test_error.mean()
        self.log_dict(metric_dict, prog_bar=False)

        try:
            if not self.logger.debug:
                hparams_log = {}
                for key, val in self.hparams.items():
                    if type(val) == list:
                        hparams_log[key] = torch.tensor(val)
                    elif isinstance(val, pl.LightningModule):
                        hparams_log[key] = val.name
                    else:
                        hparams_log[key] = val
                self.logger.experiment.add_hparams(hparams_log, metric_dict)
        except:
            print("Failed to add hparams")

    def parameters(self):
        if self.finetune_backbone:
            return chain(self.backbone.parameters(), self.task_head.parameters())
        else:
            return self.task_head.parameters()

    def configure_optimizers(self):
        if self.optim == 'SGD':
            optimizer = torch.optim.SGD(
                self.parameters(),
                lr=self.learning_rate,
                nesterov=self.nesterov,
                momentum=0.9,
                weight_decay=self.weight_decay,
            )
        elif self.optim == 'Adam':
            optimizer = torch.optim.Adam(self.parameters(),
                                         lr=self.learning_rate,
                                         weight_decay=self.weight_decay)
        else:
            raise NotImplementedError('{} optimizer is not implemented.'.format(self.optim))
        # set scheduler
        if self.scheduler_type == "step":
            scheduler_config = {
                'scheduler': torch.optim.lr_scheduler.MultiStepLR(optimizer, self.decay_epochs, gamma=self.gamma),
                'monitor': 'val_loss',
                'interval': 'step', 'frequency': self.val_steps, 'strict': True}
        elif self.scheduler_type == "cosine":
            scheduler_config = {
                'scheduler': torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer,
                    self.train_steps // self.val_steps,
                    eta_min=self.final_lr  # total epochs to run
                ),
                'monitor': 'val_loss',
                'interval': 'step', 'frequency': self.val_steps, 'strict': True}

        elif self.scheduler_type == "plateau":
            scheduler_config = {
                'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer,
                    factor=self.gamma
                ),
                'monitor': 'val_loss',
                'interval': 'step', 'frequency': self.val_steps, 'strict': True}
        elif self.scheduler_type == 'warmup_cosine':
            scheduler_config = {
                "scheduler": pl_bolts.optimizers.lr_scheduler.LinearWarmupCosineAnnealingLR(
                    optimizer, warmup_epochs=self.warmup_epochs * self.val_steps, max_epochs=self.train_steps,
                    warmup_start_lr=1e-8
                ),
                "interval": "step",
                "frequency": 1,
                "strict": True,
            }
        else:
            return {'optimizer': optimizer}

        return {'optimizer': optimizer, 'lr_scheduler': scheduler_config}

    @property
    def name(self) -> str:
        """
        Get the name of the model according its parameters
        """
        return self.__get_name()

    def __get_name(self):
        return "{}{}{}{}_{}_{}{}_lr{}_{}wd{}_{}sch_gm{}_{}-{}steps".format(self.__class__.__name__,
                                                                           '_fullytune' if self.finetune_backbone else '',
                                                                             f'_{self.mode}',
                                                                           f'_{self.n_train}train' if self.n_train is not None else '',
                                                                           'linear' if self.hidden_dim is None else f'mlp{self.hidden_dim}',
                                                                             self.in_features,
                                                                           '_sampling' if self.sampling else '',
                                                                             self.learning_rate,
                                                                             self.optim + '_' if self.optim != 'SGD' else '',
                                                                             self.weight_decay,
                                                                             self.scheduler_type,
                                                                             self.gamma,
                                                                             self.train_steps,
                                                                             self.val_steps
                                                                            )

