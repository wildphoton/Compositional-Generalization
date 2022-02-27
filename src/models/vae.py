#!/usr/bin/env python
"""
Created by zhenlinx on 10/22/2020
"""
import os
import sys
sys.path.append(os.path.realpath('..'))
from math import sqrt
import torch
from torch import nn
from torch.nn import functional as F
import torchvision.utils as vutils

from .types_ import *
import pytorch_lightning as pl
from pytorch_lightning.loggers import TestTubeLogger, NeptuneLogger, WandbLogger

from architectures import build_cnn
from evaluation.dci import DCIMetrics
from .optimizer import init_optimizer, init_lr_scheduler


class VAE(pl.LightningModule):
    def __init__(self,
                 input_size: List,
                 architecture: str,
                 latent_dim: int,
                 beta: float = 1.0,
                 recon_loss: str = 'mse',
                 lr: float = 0.001,
                 optim: str = 'adam',
                 weight_decay: float = 0,
                 **kwargs) -> None:
        """

        :param input_size: (n_channels, image_height, image_width)
        :param architecture:
        :param latent_dim:
        :param img_size:
        :param beta:
        :param recon_loss: 'mse' for Gaussian decoder and 'bce' for the bernoulli decoder
        :param lr:
        :param optim:
        :param weight_decay:
        :param kwargs:
        """

        super(VAE, self).__init__()

        self.save_hyperparameters()
        self.latent_dim = latent_dim
        self.architecture = architecture
        self.input_size = input_size
        self.beta = beta
        self.recon_loss = recon_loss

        self.optim = optim
        self.lr = lr
        self.weight_decay = weight_decay
        self.setup_models()
        pass

    def setup_models(self):
        self.encoder_conv, self.encoder_latent, self.decoder_latent, self.decoder_conv = build_cnn(
            self.input_size, self.architecture, self.latent_dim)

    def encode(self, input: Tensor) -> List[Tensor]:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) mean and variation logits of latent distribution
        """
        emb = self.encoder_conv(input)
        latent = self.encoder_latent(emb)
        mu, log_var = torch.split(latent, self.latent_dim, dim=1)
        return mu, log_var

    def decode(self, z: Tensor) -> Tensor:
        """
        Maps the given latent codes
        onto the image space.
        :param z: (Tensor) [B x D]
        :return: (Tensor) [B x C x H x W]
        """
        return self.decoder_conv(self.decoder_latent(z))

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        std = torch.exp(0.5 * logvar)
        # prior
        p = torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(std))

        try:
            q = torch.distributions.Normal(mu, std) # posterior
        except:
            print(mu, std)
        return {
            'prior': p,
            'posterior': q,
            'sampled_z': q.rsample()
        }

    def forward(self, input: Tensor, **kwargs) -> List[Tensor]:
        mu, log_var = self.encode(input)
        results = self.reparameterize(mu, log_var)
        results['recon'] = self.decode(results['sampled_z'])
        results['mu'] = mu
        results['log_var'] = log_var
        return results

    def embed(self, x, mode, sampling=True, **kwargs):
        if mode == 'pre':
            return self.encoder_conv(x)
        else:
            mu, log_var = self.encode(x)
            if sampling and self.training:
                z = self.reparameterize(mu, log_var)['sampled_z']
            else:
                z = mu
            if mode == 'latent':
                return z
            elif mode == 'post':
                return self.decoder_latent(z)
            else:
                raise ValueError()

    def compute_loss(self, inputs, results, labels=None):
        if self.recon_loss == 'mse':
            recon_loss = F.mse_loss(results['recon'], inputs, reduction='sum') / inputs.size(0)
        elif self.recon_loss == 'bce':
            recon_loss = F.binary_cross_entropy_with_logits(results['recon'], inputs, reduction='sum') / inputs.size(0)

        # p, q, z = results['prior'], results['posterior'], results['sampled_z']
        # # kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)
        # log_qz = q.log_prob(z)
        # log_pz = p.log_prob(z)
        #
        # kl = log_qz - log_pz
        mu = results['mu']
        log_var = results['log_var']
        kl = -0.5 * (1 + log_var - mu ** 2 - log_var.exp()).sum(1).mean()

        loss = recon_loss
        if self.beta > 0:
            loss += kl * self.beta
        loss_dict = {'loss': loss, 'recon_loss': recon_loss, 'kl_loss': kl}
        return loss_dict

    def step(self, batch, batch_idx, stage='train') -> dict:
        """
        Computes the VAE loss function.
        KL(N(\mu, \sigma), N(0, 1)) = \log \frac{1}{\sigma} + \frac{\sigma^2 + \mu^2}{2} - \frac{1}{2}
        :param args:
        :param kwargs:
        :return:
        """
        x, y = batch
        results = self.forward(x)
        loss_dict = self.compute_loss(x, results)
        log = {f"{stage}_{k}": v.detach() for k, v in loss_dict.items()}

        if batch_idx == 0 and self.logger and stage != 'test':
            self.sample_images(batch, stage=stage)

        # if stage == 'test':
        #     z = self.embed(x, mode='latent', sampling=False)
        #     log['z'] = z
        #     log['y'] = y
        return loss_dict['loss'], log

    def training_step(self, batch, batch_idx, optimizer_idx = 0):
        loss, logs = self.step(batch, batch_idx)
        self.log_dict(logs, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, logs = self.step(batch, batch_idx, 'val')
        self.log_dict(logs, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def test_step(self, batch, batch_idx):
        loss, logs = self.step(batch, batch_idx, 'test')
        return logs

    def test_epoch_end(self, outputs):
        metrics = {}
        for key in outputs[0]:
            if 'loss' in key:
                metrics['{}'.format(key)] = torch.stack([x[key] for x in outputs]).mean()
        self.log_dict({key: val.item() for key, val in metrics.items()}, prog_bar=False)

        # Zs = torch.cat([output['z'] for output in outputs], dim=0).float()
        # Ys = torch.cat([output['y'] for output in outputs], dim=0).float()
        # dci_metric = DCIMetrics(None, n_factors=Ys.size(1))
        # dci_score = dci_metric(model=None, model_zs=(Zs.numpy(), Ys.numpy()))[2]
        dci_metric = DCIMetrics(self.trainer.datamodule.train_dataloader(),
                   n_factors=self.trainer.datamodule.train_dataset.n_gen_factors)
        dci_score = dci_metric(model=self)[2]
        self.log('dci_disGen', dci_score)
        try:
            hparams_log = {}
            for key, val in self.hparams.items():
                if type(val) == list:
                    hparams_log[key] = torch.tensor(val)
            self.logger.experiment.add_hparams(hparams_log, metrics)
        except:
            print("Failed to add hparams")

    def configure_optimizers(self):
        optimizer = init_optimizer(self.optim, self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        return {'optimizer': optimizer}

    def sample(self,
               num_samples:int,
               current_device: int, **kwargs) -> Tensor:
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :param current_device: (Int) Device to run the model
        :return: (Tensor)
        """
        z = torch.randn(num_samples,
                        self.latent_dim)

        z = z.to(current_device)

        samples = self.decode(z)
        return samples

    def sample_images(self, batch, num=25, stage='train'):
        # Get sample reconstruction image
        inputs, labels = batch
        if inputs.size(0)>num:
           inputs, labels = inputs[:num], labels[:num]
        recons = self.forward(inputs, labels=labels)['recon']
        if  self.recon_loss == 'bce':
            recons = torch.sigmoid(recons)

        inputs_grids = vutils.make_grid(inputs, normalize=True, nrow=int(sqrt(num)), pad_value=1)
        recon_grids = vutils.make_grid(recons, normalize=True, nrow=int(sqrt(num)), pad_value=1)
        self.logger.log_image(key=f'input_{stage}', images=[inputs_grids],
                              caption=[f'epoch_{self.current_epoch}'])
        self.logger.log_image(key=f'recon_{stage}', images=[recon_grids],
                              caption=[f'epoch_{self.current_epoch}'])
        del inputs, recons

    def log_image(self, img_name, img, description=None):
        if isinstance(self.logger, TestTubeLogger):
            self.logger.experiment.add_image(img_name, img, self.current_epoch)
        elif isinstance(self.logger, NeptuneLogger):
            if isinstance(img, torch.Tensor) and len(img.size())==3 and img.size(0) <=3:
                img = img.permute([1,2,0]).detach().cpu()
            self.logger.experiment.log_image(img_name, x=self.current_epoch, y=img.detach().cpu())
        elif isinstance(self.logger, WandbLogger):
            self.logger.log_image(key=img_name, images=[img], caption=[description])
        else:
            raise NotImplementedError('The log image function for {} is not implemented yet'.format(type(self.logger)))

    @property
    def name(self) -> str:
        return self.make_name()

    @property
    def backbone_name(self) -> str:
        return self.make_backbone_name()

    def make_name(self) -> str:
        """
        Get the name of the model according its parameters
        """
        return "{}_{}_beta{}_{}_lr{}_{}_wd{}".format(
            self.__class__.__name__,
            self.make_backbone_name(),
            self.beta,
            self.recon_loss,
            self.lr,
            self.optim,
            self.weight_decay,
        )

    def make_backbone_name(self) -> str:
        """
        Get the name of the backbone according its parameters
        """
        return "{}_z{}".format(
            self.architecture,
            self.latent_dim,
        )

    def get_rep_size(self, mode):
        if mode == 'latent':
            return self.latent_dim
        elif mode == 'pre':
            return self.encoder_conv.output_size
        elif mode == 'post':
            return self.decoder_latent.output_size
        else:
            raise ValueError()