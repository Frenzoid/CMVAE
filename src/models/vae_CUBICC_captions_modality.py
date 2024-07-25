# Unimodal VAE CUBICC Text Modality Specification
import os
import json
import numpy as np
import torch
import torch.distributions as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from utils import Constants
from .base_vae import VAE
from .encoder_decoder_blocks.cnn_cub_text import Enc, Dec

# Constants
maxSentLen = 32  # max length of any description for birds dataset
minOccur = 3
embeddingDim = 128
lenWindow = 3
fBase = 32
vocabSize = 1590


class CUB_Sentence(VAE):
    """ Unimodal VAE subclass for Text modality CUBICC experiment """

    def __init__(self, params):
        super(CUB_Sentence, self).__init__(
            prior_dist=dist.Normal if params.priorposterior == 'Normal' else dist.Laplace,      # prior
            likelihood_dist=dist.OneHotCategorical,                                             # likelihood
            post_dist=dist.Normal if params.priorposterior == 'Normal' else dist.Laplace,       # posterior
            enc=Enc(params.latent_dim_w, params.latent_dim_z, dist=params.priorposterior),      # Encoder model
            dec=Dec(params.latent_dim_w, params.latent_dim_z),                                  # Decoder model
            params=params)                                                                      # Params (args passed to main)
        grad_w = {'requires_grad': True}
        self._pw_params_aux = nn.ParameterList([
            nn.Parameter(torch.zeros(1, params.latent_dim_w), requires_grad=False),  # mu
            nn.Parameter(torch.zeros(1, params.latent_dim_w), **grad_w)  # logvar
        ])
        self._pw_params_std = nn.ParameterList([
            nn.Parameter(torch.zeros(1, params.latent_dim_w), requires_grad=False),  # mu
            nn.Parameter(torch.zeros(1, params.latent_dim_w), requires_grad=False)  # logvar
        ])
        grad = {'requires_grad': False}
        grad_c = {'requires_grad': params.learn_prior_c}
        self._pc_params = nn.ParameterList([
            nn.Parameter(torch.zeros(1, params.latent_dim_c), **grad_c)
        ])
        self._pz_params_m = nn.ParameterList([
            nn.Parameter((((2*torch.rand(1, params.latent_dim_z))-1)/2), requires_grad=True) for c_k in range(params.latent_dim_c)
        ])
        self._pz_params_lv = nn.ParameterList([
            nn.Parameter(torch.zeros(1, params.latent_dim_z), **grad) for c_k in range(params.latent_dim_c)
        ])
        self.modelName = 'cubS_resnet'
        self.llik_scaling = 1.

        self.fn_2i = lambda t: t.cpu().numpy().astype(int)
        self.fn_trun = lambda s: s[:np.where(s == 2)[0][0] + 1] if 2 in s else s
        self.vocab_file = params.datadir + '/CUBICC/cub.vocab'

        self.maxSentLen = maxSentLen
        self.vocabSize = vocabSize

        self.i2w = self.load_vocab()
        self.params = params


    def pz_params(self, idx):
        """
        Get prior distrbution parameters for given latent cluster (indexed by idx)
        Args:
            idx: Latent cluster

        Returns:
            Prior distribution parameters
        """
        if self.params.priorposterior == 'Normal':
            return self._pz_params_m[idx],  F.softplus(self._pz_params_lv[idx]) + Constants.eta
        else:
            return self._pz_params_m[idx], F.softmax(self._pz_params_lv[idx], dim=-1) * self._pz_params_lv[idx].size(-1) + Constants.eta

    @property
    def pc_params(self):
        """

           Returns: Parameters of uniform prior distribution on latent clusters.

           """
        return F.softmax(self._pc_params[0], dim=-1)

    @property
    def pw_params_aux(self):
        """

        Returns: Parameters of prior distribution for modality-specific latent code

        """
        if self.params.priorposterior == 'Normal':
            return self._pw_params_aux[0], F.softplus(self._pw_params_aux[1]) + Constants.eta
        else:
            return self._pw_params_aux[0], F.softmax(self._pw_params_aux[1], dim=-1) * self._pw_params_aux[1].size(-1) + Constants.eta

    @property
    def pw_params_std(self):
        """

        Returns: Parameters of prior distribution for modality-specific latent code

        """
        if self.params.priorposterior == 'Normal':
            return self._pw_params_std[0], F.softplus(self._pw_params_std[1]) + Constants.eta
        else:
            return self._pw_params_std[0], F.softmax(self._pw_params_std[1], dim=-1) * self._pw_params_std[1].size(-1) + Constants.eta


    def load_vocab(self):
        # call dataloader function to create vocab file
        assert os.path.exists(self.vocab_file)
        with open(self.vocab_file, 'r') as vocab_file:
            vocab = json.load(vocab_file)
        return vocab['i2w']


    
