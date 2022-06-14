#!/usr/bin/env python
"""
Beta Total Correlation VAE
"""
import os
import sys
sys.path.append(os.path.realpath('..'))
import torch
import math
from models.vae import VAE

class BetaTCVAE(VAE):

    def __init__(self,
                 alpha=1,
                 gamma=1,
                 mss=False,
                 **kwargs):
        self.alpha = alpha
        self.gamma = gamma
        self.mss = mss
        super(BetaTCVAE, self).__init__(**kwargs)

    def compute_loss(self, inputs, results, labels=None):
        recon_loss = self.compute_recontruct_loss(inputs, results)
        mi, tc, dim_KL = self.compute_KLD_loss(results)
        loss = recon_loss + self.alpha * mi + self.beta * tc + self.gamma * dim_KL
        loss_dict = {'loss': loss, 'recon_loss': recon_loss,
                     'mi': mi, 'tc': tc, 'dim_KL': dim_KL}
        return loss_dict

    def compute_KLD_loss(self, results):
        """Compute decomposed KL loss"""
        mu = results['mu']
        log_var = results['log_var']
        z = results['z']
        batch_size, dim = z.shape
        try:
            dataset_size = len(self.trainer.datamodule.train_dataset)
        except:
            dataset_size = 10000

        log_pz = gaussian_log_density(z, torch.zeros_like(z), torch.zeros_like(z)).sum(1)
        log_qz_cond_x = gaussian_log_density(z, mu, log_var).sum(1)

        # compute log q(z) ~= log 1/(NM) sum_m=1^M q(z|x_m) = - log(MN) + logsumexp_m(q(z|x_m))
        matrix_log_qz = gaussian_log_density(z.view(batch_size, 1, dim),
                                  mu.view(1, batch_size, dim),
                                  log_var.view(1, batch_size, dim))

        if not self.mss:
            # minibatch weighted sampling
            log_qz_prod_marginals = (torch.logsumexp(matrix_log_qz, dim=1, keepdim=False) - math.log(batch_size * dataset_size)).sum(1)
            log_qz = (torch.logsumexp(matrix_log_qz.sum(2), dim=1, keepdim=False) - math.log(batch_size * dataset_size))
        else:
            # minibatch stratified sampling
            logiw_matrix = log_importance_weight_matrix(batch_size, dataset_size).type_as(matrix_log_qz.data)
            log_qz = torch.logsumexp(logiw_matrix + matrix_log_qz.sum(2), dim=1, keepdim=False)
            log_qz_prod_marginals = torch.logsumexp(
                logiw_matrix.view(batch_size, batch_size, 1) + matrix_log_qz, dim=1, keepdim=False).sum(1)

        return (log_qz_cond_x - log_qz).mean(), (log_qz - log_qz_prod_marginals).mean(), (log_qz_prod_marginals - log_pz).mean()

    def make_name(self) -> str:
        """
        Get the name of the model according its parameters
        """
        return "{}_{}_alpha{}beta{}gamma{}{}_{}_lr{}_{}_wd{}".format(
            self.__class__.__name__,
            self.make_backbone_name(),
            self.alpha,
            self.beta,
            self.gamma,
            '_mss' if self.mss else '',
            self.recon_loss,
            self.lr,
            self.optim,
            self.weight_decay,
        )

def gaussian_log_density(samples, mean, log_var):
    """ Estimate the log density of a Gaussian distribution
    Borrowed from https://github.com/google-research/disentanglement_lib/
    :param samples: batched samples of the Gaussian densities with mean=mean and log of variance = log_var
    :param mean: batched means of Gaussian densities
    :param log_var: batches means of log_vars
    :return:
    """
    import math
    pi = torch.tensor(math.pi, requires_grad=False)
    normalization = torch.log(2. * pi)
    inv_var = torch.exp(-log_var)
    tmp = samples - mean
    return -0.5 * (tmp * tmp * inv_var + log_var + normalization)


def matrix_log_density_gaussian(x, mu, logvar):
    """Calculates log density of a Gaussian for all combination of bacth pairs of
    `x` and `mu`. I.e. return tensor of shape `(batch_size, batch_size, dim)`
    instead of (batch_size, dim) in the usual log density.
    Parameters
    ----------
    x: torch.Tensor
        Value at which to compute the density. Shape: (batch_size, dim).
    mu: torch.Tensor
        Mean. Shape: (batch_size, dim).
    logvar: torch.Tensor
        Log variance. Shape: (batch_size, dim).
    batch_size: int
        number of training images in the batch
    """
    batch_size, dim = x.shape
    x = x.view(batch_size, 1, dim)
    mu = mu.view(1, batch_size, dim)
    logvar = logvar.view(1, batch_size, dim)
    return gaussian_log_density(x, mu, logvar)

def log_importance_weight_matrix(batch_size, dataset_size):
    N = dataset_size
    M = batch_size - 1
    strat_weight = (N - M) / (N * M)
    W = torch.Tensor(batch_size, batch_size).fill_(1 / M)
    W.view(-1)[::M + 1] = 1 / N
    W.view(-1)[1::M + 1] = strat_weight
    W[M - 1, 0] = strat_weight
    return W.log()

