# CUBICC Unimodal VAE Image model specification
import torch
import torch.distributions as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from utils import Constants
from .base_vae import VAE
from .encoder_decoder_blocks.resnet_cub_image import EncoderImg, DecoderImg


#############################################################################################
class CUB_Image(VAE):
    """ Unimodal VAE subclass for Image modality CUBICC experiment """

    def __init__(self, params):
        super(CUB_Image, self).__init__(
            dist.Normal if params.priorposterior == 'Normal' else dist.Laplace,                 # prior
            dist.Laplace,                                                                       # likelihood
            dist.Normal if params.priorposterior == 'Normal' else dist.Laplace,                 # posterior
            EncoderImg(params.latent_dim_w, params.latent_dim_z, dist=params.priorposterior),   # Encoder model
            DecoderImg(params.latent_dim_w + params.latent_dim_z),                              # Decoder model
            params                                                                              # Params (args passed to main)
        )
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
        self._pu_params_m = nn.ParameterList([
            nn.Parameter((((2 * torch.rand(1, params.latent_dim_z)) - 1) / 2), requires_grad=True) for c_k in
            range(params.latent_dim_c)
        ])
        self._pu_params_lv = nn.ParameterList([
            nn.Parameter(torch.zeros(1, params.latent_dim_z), **grad) for c_k in range(params.latent_dim_c)
        ])
        self.modelName = 'cubI'
        self.dataSize = torch.Size([3, 64, 64])
        self.llik_scaling = 1.
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
            return self._pz_params_m[idx], F.softplus(self._pz_params_lv[idx]) + Constants.eta
        else:
            return self._pz_params_m[idx], F.softmax(self._pz_params_lv[idx], dim=-1) * self._pz_params_lv[idx].size(
                -1) + Constants.eta

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
            return self._pw_params_aux[0], F.softmax(self._pw_params_aux[1], dim=-1) * self._pw_params_aux[1].size(
                -1) + Constants.eta

    @property
    def pw_params_std(self):
        """

        Returns: Parameters of prior distribution for modality-specific latent code

        """
        if self.params.priorposterior == 'Normal':
            return self._pw_params_std[0], F.softplus(self._pw_params_std[1]) + Constants.eta
        else:
            return self._pw_params_std[0], F.softmax(self._pw_params_std[1], dim=-1) * self._pw_params_std[1].size(
                -1) + Constants.eta


