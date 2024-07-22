# PolyMNIST model specification
import torch
import torch.distributions as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image, make_grid
from utils import Constants
from .base_vae import VAE
from datasets_PolyMNIST import PolyMNISTDataset
from .encoder_decoder_blocks.resnet_polyMNIST import Enc, Dec

# Constants
dataSize = torch.Size([3, 28, 28])

class PolyMNIST(VAE):
    """ Unimodal VAE subclass for PolyMNIST experiment """

    def __init__(self, params):
        super(PolyMNIST, self).__init__(
            dist.Normal if params.priorposterior == 'Normal' else dist.Laplace,         # prior
            dist.Laplace,  # likelihood
            dist.Normal if params.priorposterior == 'Normal' else dist.Laplace,         # posterior
            Enc(params.latent_dim_w, params.latent_dim_z, dist=params.priorposterior),  # Encoder model
            Dec(params.latent_dim_w+params.latent_dim_z),                               # Decoder model
            params                                                                      # params (args from main)
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
        self._pz_params_m = nn.ParameterList([
            nn.Parameter((((2*torch.rand(1, params.latent_dim_z))-1)/2), requires_grad=True) for c_k in range(params.latent_dim_c)
        ])
        self._pz_params_lv = nn.ParameterList([
            nn.Parameter(torch.zeros(1, params.latent_dim_z), **grad) for c_k in range(params.latent_dim_c)
        ])
        self.modelName = 'polymnist-split'
        self.dataSize = dataSize
        self.llik_scaling = 1.
        self.datadir = params.datadir
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


    def getDataLoaders(self, batch_size, shuffle=True, device='cuda'):
        """Get PolyMNIST modality dataloaders."""
        unim_datapaths_train = [self.datadir+"/PolyMNIST/train/" + "m" + str(self.modal)]
        unim_datapaths_test = [self.datadir+"/PolyMNIST/test/" + "m" + str(self.modal)]
        
        kwargs = {'num_workers': 1, 'pin_memory': True} if device == 'cuda' else {}
        tx = transforms.ToTensor()
        train = DataLoader(PolyMNISTDataset(unim_datapaths_train, transform=tx),
                           batch_size=batch_size, shuffle=shuffle, **kwargs)
        test = DataLoader(PolyMNISTDataset(unim_datapaths_test, transform=tx),
                           batch_size=batch_size, shuffle=shuffle, **kwargs)
        return train, test


    def generate_unconditional_random_to_tensor(self, N):
        """
        Unconditional random generation.
        Returns:
                Tensor of unconditional random generations.
        """
        samples = super(PolyMNIST, self).generate_unconditional_random(N)
        samples = samples.data.cpu()

        return make_grid(samples, nrow=N)

    def generate_unconditional_to_tensor(self, N):
        """
        Unconditional generation.
        Returns:
                Tensor of unconditional generations.
        """
        samples = super(PolyMNIST, self).generate_unconditional(N)
        samples = samples.data.cpu()

        return make_grid(samples, nrow=N)

    def generate_unconditional_random_for_fid_calculation(self, savePath, num_samples, tranche):
        """
        Unconditional random generation for FID calculation. (Split in tranches for memory issues)
        Args:
            savePath: Path of directory where to save images
            num_samples: Num_samples to generate
            tranche: Tranche of images currently generated

        """
        N = num_samples
        samples = super(PolyMNIST, self).generate_unconditional_random(N)
        samples = samples.data.cpu()
        for image in range(samples.size(0)):
            save_image(samples[image, :, :, :], '{}/random/m{}/{}_{}.png'.format(savePath, self.modal, tranche, image))

    def self_and_cross_modal_reconstruct_for_fid(self, data, savePath, i):
        """
        Conditional generation for FID calculation.
        Args:
            data: input
            savePath: Path of directory where to save images
            i: index naming

        """
        recon = super(PolyMNIST, self).reconstruct(data)
        recon = recon.squeeze(0).cpu()
        for image in range(recon.size(0)):
            save_image(recon[image, :, :, :], '{}/m{}/m{}/{}_{}.png'.format(savePath, self.modal,self.modal, image, i))


    def reconstruct_to_tensor(self, data, N=10):
        """
        Test-time reconstruction.
        Returns:
                Tensor of reconstructions.
        """
        recon = super(PolyMNIST, self).reconstruct(data[:N])
        recon = recon.squeeze(0)
        comp = torch.cat([data[:N], recon]).data.cpu()
        return make_grid(comp, nrow=N)
