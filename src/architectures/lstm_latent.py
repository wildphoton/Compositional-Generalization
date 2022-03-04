#!/usr/bin/env python
"""
Latent layers using LSTM

Created by zhenlinx on 03/03/2022
"""
import torch
import torch.nn as nn
from architectures.stochastic import gumble_softmax, straight_through_discretize

class LatentModuleLSTM(nn.Module):
    """
    A module that encode a feature vector as a sequence of discrete tokens autoregressively
    with LSTM and decode the sequence into a feature vector with another LSTM.
    """
    def __init__(self,
                 input_size,
                 output_size,
                 hidden_size,
                 latent_size,
                 dictionary_size,
                 fix_length=False,
                 temperature=1.0,
                 **kwargs):
        super(LatentModuleLSTM, self).__init__(**kwargs)
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.latent_size = latent_size
        self.dictionary_size = dictionary_size
        self.temperature = temperature
        self.fix_length = fix_length

        self.input_layer = nn.Linear(self.input_size, self.hidden_size)
        self.output_layer = nn.Linear(self.hidden_size, self.output_size)
        self.encoder_lstm = nn.LSTMCell(self.hidden_size, self.hidden_size)
        self.decoder_lstm = nn.LSTMCell(self.hidden_size, self.hidden_size)
        self.hidden_to_token = nn.Linear(self.hidden_size,
                                         self.dictionary_size)
        self.token_to_hidden = nn.Linear(self.dictionary_size, self.hidden_size)

        if not fix_length:
            # if using variable length, token #0 is the eos token and indicate the end of sequence
            self.register_buffer('eos_token',
                                 torch.zeros(1, self.dictionary_size).scatter_(-1, torch.tensor([[0, ]]), 1))

    def forward(self, input, sampling=True, **kwargs):
        res = self.encode(input, sampling)
        res['output'] = self.decode(**res)
        return res

    def encode(self, x, sampling):
        """
        Encode a batch of feature vector into a batch of sequences of discrete tokens
        :param x: (Tensor) input feature  [B x input_size]
        :return:
            z: (Tensor) sequence of discrete tokens in one-hot shapes [B, latent_size, dictionary_size]
            logits: (Tensor) sequence of logits from which tokens were sampled from [B, latent_size, dictionary_size]
        """
        x = self.input_layer(x)
        if self.fix_length:
            raise NotImplementedError
        else:
            tokens_one_hot, logits, eos_id = self.encode_variable_length(x, sampling=sampling)
            res = {
                'z': tokens_one_hot,
                'logits': logits,
                'eos_id': eos_id,
            }
        return res

    def decode(self, z, eos_id=None, **kwargs):
        if self.fix_length:
            raise NotImplementedError
        else:
            output = self.decode_variable_length(z, eos_id)
        output = self.output_layer(output)
        return output


    def encode_variable_length(self, x, sampling=True):
        """
        transform an feature into a sequence of discrate tokens (as one-hot vectors)
        :param x: image feature with size of self.hidden_size
        :param sampling: if True, using Gumble-softmax to sample tokens from distributions,
            otherwise use the token with the highest probability.
        :return:
        """
        _device = x.device
        samples = []
        logits = []
        batch_size = x.shape[0]
        hx = torch.zeros(batch_size, self.hidden_size,
                         device=_device)
        cx = x

        is_finished = torch.zeros(batch_size, device=_device).bool()

        eos_ind = torch.zeros(batch_size, device=_device, dtype=torch.long)  # the index where the first eos appears
        lstm_input = torch.zeros(batch_size, self.hidden_size,
                                 device=_device)

        eos_batch = self.eos_token.to(_device).repeat(batch_size, 1)

        for num in range(self.latent_size):
            hx, cx = self.encoder_lstm(lstm_input, (hx, cx))
            pre_logits = self.hidden_to_token(hx)  # embedding to catogory logits
            logits.append(pre_logits)

            if sampling and self.training:
                # sample discrete code with gumble softmax
                z_sampled_soft = gumble_softmax(pre_logits, self.temperature)
            else:
                z_sampled_soft = torch.softmax(pre_logits, dim=-1)

            z_sampled_onehot, z_argmax = straight_through_discretize(z_sampled_soft)

            z_sampled_onehot[is_finished] = eos_batch[is_finished]
            # z_one_hot = z_sampled_onehot
            samples.append(z_sampled_onehot)

            # the projected embedding of the sampled discrete code is the input for the next step
            lstm_input = self.token_to_hidden(z_sampled_onehot)

            # record ending state of this step
            is_finished = torch.logical_or(is_finished, z_argmax == 0)
            # not_finished = (not_finished * (z_argmax != 0)).bool()
            not_finished = torch.logical_not(is_finished)
            eos_ind += not_finished.long()

        logits = torch.stack(logits).permute(1, 0, 2)
        samples = torch.stack(samples).permute(1, 0, 2)
        return samples, logits, eos_ind


    def decode_variable_length(self, z, eos_ind):
        """

        :param z: onehot discrete representions (BatchSize x LatentCodeSize x VacabularySize )
        :param eos_ind: index of EOS token for latent z (BatchSize) e.g. if eos_ind == 2, z[0:2] are meaningfull tokens
        :return:
        """
        batch_size = z.shape[0]
        _device = z.device

        z_embeddings = self.token_to_hidden(z.contiguous().view(-1, z.shape[-1])).view(batch_size, self.latent_size, -1)  # project one-hot codes into continueious embeddings
        hx = torch.zeros(batch_size, self.hidden_size,
                         device=_device)
        cx = torch.zeros(batch_size, self.hidden_size,
                         device=_device)
        outputs = []
        for n in range(self.latent_size):
            inputs = z_embeddings[:,n]
            hx, cx = self.decoder_lstm(inputs, (hx, cx))
            outputs.append(hx)

        # we also feed EOS embeddings to decoder LSTM
        eos_embeddings = self.token_to_hidden(self.eos_token.to(_device).repeat(batch_size, 1))
        hx, cx = self.decoder_lstm(eos_embeddings, (hx, cx))
        outputs.append(hx)

        outputs = torch.stack(outputs).permute(1, 0, 2)

        # select right output according to the EOS position in the latent code sequence.
        # mask_ind = eos_ind
        eos_ind_mask = torch.zeros(batch_size, self.latent_size + 1, 1, device=_device).scatter_(1, eos_ind.view(-1, 1, 1), 1)
        selected_output = outputs.masked_select(eos_ind_mask.bool()).view(batch_size, -1)
        return selected_output