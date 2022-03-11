#!/usr/bin/env python
"""
Created by zhenlinx on 10/22/2020
"""
# sys.path.append(os.path.realpath('..'))
from math import sqrt
import torch
from torch.nn import functional as F
import torchvision.utils as vutils

import pytorch_lightning as pl

from architectures.helper import build_architectures
from evaluation.dci import DCIMetrics
from .optimizer import init_optimizer
from models.types_ import *


class VAE(pl.LightningModule):
    def __init__(self,
                 input_size: List,
                 architecture: str,
                 latent_size: int,
                 beta: float = 1.0,
                 icc: bool = False,
                 icc_max: float = 20,
                 icc_min: float = 0,
                 icc_steps: float = 10000,
                 recon_loss: str = 'mse',
                 lr: float = 0.001,
                 optim: str = 'adam',
                 weight_decay: float = 0,
                 **kwargs) -> None:
        """

        :param input_size: (n_channels, image_height, image_width)
        :param architecture:
        :param latent_size:
        :param img_size:
        :param beta:
        :param icc: |KL - C|*beta with a increasing schedule on C based on: Understanding disentangling in β-VAE  https://arxiv.org/pdf/1804.03599.pdf
        :param recon_loss: 'mse' for Gaussian decoder and 'bce' for the bernoulli decoder
        :param lr:
        :param optim:
        :param weight_decay:
        :param kwargs:
        """

        super(VAE, self).__init__()

        self.save_hyperparameters()
        self.latent_size = latent_size
        self.architecture = architecture
        self.input_size = input_size
        self.beta = beta
        self.recon_loss = recon_loss

        # information capacity control hyperparameters
        # in original paper, icc is from 0 to 25 nats, icc_step = 100000 iters, beta=1000
        self.icc = icc
        self.icc_max = icc_max
        self.icc_min = icc_min
        self.icc_steps = icc_steps

        self.optim = optim
        self.lr = lr
        self.weight_decay = weight_decay
        self.setup_models()
        pass

    def setup_models(self):
        (self.encoder_conv, self.decoder_conv), (self.encoder_latent, self.decoder_latent) = build_architectures(
            self.input_size, self.architecture, self.latent_size, model=self.__class__.__name__)

    def encode_latent(self, feat, sampling):
        """
        Encode the feature from backbone to latent variables and reparameterize it
        :param emb:
        :param sampling: (Bool) if sample from latent distribution
        :return:
        """
        latent = self.encoder_latent(feat)
        results = self.reparameterize(latent, sampling=sampling)
        return results

    def decode_latent(self, z):
        """
        Decode latent variable z into a feature
        :param z:
        :return:
        """
        return self.decoder_latent(z)

    def encode(self, input: Tensor, sampling) -> List[Tensor]:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :param sampling: (Bool) if sample from latent distribution
        :return: (Tensor) mean and variation logits of latent distribution
        """
        feat = self.encoder_conv(input)
        res = self.encode_latent(feat, sampling=sampling)
        return res

    def decode(self, inputs: Tensor) -> Tensor:
        """
        Maps the given latent codes
        onto the image space.
        :param inputs: A dictionary of encoder outputs
        :return: (Tensor) [B x C x H x W]
        """
        return self.decoder_conv(self.decode_latent(inputs['z']))

    def reparameterize(self, latent: dict, sampling) -> dict:
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param Input dictionary with two keys:
            mu: (Tensor) Mean of the latent Gaussian [B x D]
            logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :param sampling: (Bool) if sample from latent distribution
        :return: output dictionary
        """
        mu, log_var = torch.split(latent, self.latent_size, dim=1)
        std = torch.exp(0.5 * log_var)
        # prior
        p = torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(std))

        try:
            q = torch.distributions.Normal(mu, std) # posterior
        except:
            print(mu, std)
        return {
            'mu': mu,
            'log_var': log_var,
            'prior': p,
            'posterior': q,
            'z': q.rsample() if sampling else mu,
        }

    def forward(self, input: Tensor, **kwargs) -> List[Tensor]:
        results = self.encode(input, sampling=self.training)
        results['recon'] = self.decode(results)
        return results

    def embed(self, x, mode, sampling=False, **kwargs):
        """
        Function to call to use VAE as a backbone model for downstream tasks
        :param x:
        :param mode:
        :param sampling:
        :param kwargs:
        :return:
        """
        if mode == 'pre':
            return self.encoder_conv(x)
        else:
            enc_res = self.encode(x, sampling=sampling)
            if mode == 'latent':
                return enc_res['z']
            elif mode == 'post':
                return self.decoder_latent(enc_res['z'])
            else:
                raise ValueError()

    def compute_loss(self, inputs, results, labels=None):
        if self.recon_loss == 'mse':
            recon_loss = F.mse_loss(results['recon'], inputs, reduction='sum') / inputs.size(0)
        elif self.recon_loss == 'bce':
            recon_loss = F.binary_cross_entropy_with_logits(results['recon'], inputs, reduction='sum') / inputs.size(0)

        loss = recon_loss

        kld_loss = self.compute_KLD_loss(results)

        if self.beta > 0:
            if self.icc:
                capacity = min(self.icc_max,
                               self.icc_min + (self.icc_max - self.icc_min) * self.global_step / self.icc_steps)
                loss += (kld_loss - capacity).abs() * self.beta
            else:
                loss += kld_loss * self.beta
        loss_dict = {'loss': loss, 'recon_loss': recon_loss, 'kl_loss': kld_loss}
        return loss_dict

    def compute_KLD_loss(self, results):
        mu = results['mu']
        log_var = results['log_var']
        kl = -0.5 * ((1 + log_var - mu ** 2 - log_var.exp()).sum(1).mean(0))
        return kl

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
        return "{}_{}_beta{}{}_{}_lr{}_{}_wd{}".format(
            self.__class__.__name__,
            self.make_backbone_name(),
            self.beta,
            f'_icc{self.icc_min}-{self.icc_max}-{self.icc_steps}steps' if self.icc else '',
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
            self.latent_size,
        )

    def get_rep_size(self, mode):
        if mode == 'latent':
            return self.latent_size
        elif mode == 'pre':
            return self.encoder_conv.output_size
        elif mode == 'post':
            return self.decoder_latent.output_size
        else:
            raise ValueError()