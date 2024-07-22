# PolyMNIST experiment CMVAE model specifications
import torch
import torch.distributions as dist
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import save_image, make_grid
from torchvision import transforms
from datasets_PolyMNIST import PolyMNISTDataset
from utils import Constants
from .cmvae import CMVAE
from .vae_polyMNIST_single_modality import PolyMNIST
from numpy import sqrt



class PolyMNIST_5modalities(CMVAE):
    """
    CMVAE subclass for PolyMNIST Experiment
    """
    def __init__(self, params):
        super(PolyMNIST_5modalities, self).__init__(dist.Normal if params.priorposterior == 'Normal' else dist.Laplace, params, PolyMNIST, PolyMNIST, PolyMNIST, PolyMNIST, PolyMNIST)

        grad = {'requires_grad': False}
        grad_c = {'requires_grad': params.learn_prior_c}
        self._pc_params = nn.ParameterList([
            nn.Parameter(torch.zeros(1, params.latent_dim_c), **grad_c)
        ])
        self._pz_params_m = nn.ParameterList([
            nn.Parameter(((2*torch.rand(1, params.latent_dim_z))-1), requires_grad=True) for c_k in range(params.latent_dim_c)
        ])
        self._pz_params_lv = nn.ParameterList([
            nn.Parameter(torch.zeros(1, params.latent_dim_z), **grad) for c_k in range(params.latent_dim_c)
        ])
        self._pw_params = nn.ParameterList([
            nn.Parameter(torch.zeros(1, params.latent_dim_w), requires_grad=False),  # mu
            nn.Parameter(torch.zeros(1, params.latent_dim_w), requires_grad=False)  # logvar
        ])
        self.modelName = 'CMVAE_PolyMNIST'

        # Fix model names for indiviudal models to be saved
        for idx, vae in enumerate(self.vaes):
            vae.modelName = 'VAE_PolyMNIST_m' + str(idx)

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


    def pc_params_pruning(self, idxs_to_prune):
        """
        Get prior probabilities of latent clusters when some clusters are pruned (setting prior probability to zero)
        Args:
            idxs_to_anneal: latent clusters to prune

        Returns: Parameters of prior distribution on latent clusters.

        """
        pc_a = self._pc_params[0].clone().detach()
        for idx_to_anneal in idxs_to_prune:
            pc_a[:, idx_to_anneal] = -100000000.0
        return F.softmax(pc_a, dim=-1)


    @property
    def pw_params(self):
        """

        Returns: Parameters of prior distribution for modality-specific latent code

        """
        if self.params.priorposterior == 'Normal':
            return self._pw_params[0], F.softplus(self._pw_params[1]) + Constants.eta
        else:
            return self._pw_params[0], F.softmax(self._pw_params[1], dim=-1) * self._pw_params[1].size(-1) + Constants.eta


    def getDataSets(self, batch_size, shuffle=True, device='cuda'):
        """Get PolyMNIST datasets."""
        tx = transforms.ToTensor()
        unim_train_datapaths = [self.datadir+"/PolyMNIST/train/" + "m" + str(i) for i in [0, 1, 2, 3, 4]]
        unim_test_datapaths = [self.datadir+"/PolyMNIST/test/" + "m" + str(i) for i in [0, 1, 2, 3, 4]]
        dataset_PolyMNIST_train = PolyMNISTDataset(unim_train_datapaths, transform=tx)
        dataset_PolyMNIST_test = PolyMNISTDataset(unim_test_datapaths, transform=tx)
        # kwargs = {'num_workers': 2, 'pin_memory': True} if device == 'cuda' else {}
        # train = DataLoader(dataset_PolyMNIST_train, batch_size=batch_size, shuffle=shuffle, **kwargs)
        # test = DataLoader(dataset_PolyMNIST_test, batch_size=batch_size, shuffle=shuffle, **kwargs)
        return dataset_PolyMNIST_train, dataset_PolyMNIST_test

    def generate_unconditional(self, N=100, indexes_to_prune=None, indexes_to_select=None, random=False, coherence_calculation=False, fid_calculation=False, savePath=None, tranche=None):
        """
        Generate unconditional
        Args:
            N: Number of generations.
            indexes_to_prune: Latent clusters to prune
            indexes_to_select: Latent clusters to select
            random: Randomly sampling latent clusters or not
            coherence_calculation: Whether it serves for coherence calculation
            fid_calculation: Whether it serves for fid calculation
            savePath: additional argument for fid calculation save path for images
            tranche: argument needed for naming of images when saved for fid calculation

        Returns:

        """
        outputs = []
        if indexes_to_prune is None and indexes_to_select is None and random == True:
            samples_list = super(PolyMNIST_5modalities, self).generate_random_unconditional(N)
        if indexes_to_prune is None and indexes_to_select is None and random == False:
            samples_list = super(PolyMNIST_5modalities, self).generate_unconditional(N)
        if indexes_to_prune is not None and indexes_to_select is None and random == True:
            samples_list = super(PolyMNIST_5modalities, self).generate_random_unconditional_with_pruning(N, indexes_to_prune)
        if indexes_to_prune is None and indexes_to_select is not None and random == False:
            samples_list = super(PolyMNIST_5modalities, self).generate_unconditional_with_input_latent_clusters(int(sqrt(N)), indexes_to_select)
        if coherence_calculation:
            return [samples.data.cpu() for samples in samples_list]
        elif fid_calculation:
            for i, samples in enumerate(samples_list):
                samples = samples.data.cpu()
                # wrangle things so they come out tiled
                # samples = samples.view(N, *samples.size()[1:])
                for image in range(samples.size(0)):
                    save_image(samples[image, :, :, :], '{}/random/m{}/{}_{}.png'.format(savePath, i, tranche, image))
        else:
            for i, samples in enumerate(samples_list):
                samples = samples.data.cpu()
                samples = samples.view(samples.size()[0], *samples.size()[1:])
                outputs.append(make_grid(samples, nrow=int(sqrt(N))))

        return outputs


    def self_and_cross_modal_generation_for_fid_calculation(self, data, savePath, i):
        """
        Self- and cross-modal reconstruction for FID calculation.
        Args:
            data: Input
            savePath: Save path
            i: image index for naming
        """
        recons_mat = super(PolyMNIST_5modalities, self).self_and_cross_modal_generation([d for d in data])
        for r, recons_list in enumerate(recons_mat):
            for o, recon in enumerate(recons_list):
                recon = recon.squeeze(0).cpu()

                for image in range(recon.size(0)):
                    save_image(recon[image, :, :, :],
                                '{}/m{}/m{}/{}_{}.png'.format(savePath, r,o, image, i))

    def self_and_cross_modal_generation(self, data, num=10,N=10):
        """
        Self- and cross-modal generation.
        Args:
            data: Input
            num: Number of samples to be considered for generation
            N: Number of generations

        Returns:
            Generations
        """
        recon_triess = [[[] for i in range(num)] for j in range(num)]
        outputss = [[[] for i in range(num)] for j in range(num)]
        for i in range(N):
            recons_mat = super(PolyMNIST_5modalities, self).self_and_cross_modal_generation([d[:num] for d in data])
            for r, recons_list in enumerate(recons_mat):
                for o, recon in enumerate(recons_list):
                      recon = recon.squeeze(0).cpu()
                      recon_triess[r][o].append(recon)
        for r, recons_list in enumerate(recons_mat):
            for o, recon in enumerate(recons_list):
                outputss[r][o] = make_grid(torch.cat([data[r][:num].cpu()]+recon_triess[r][o]), nrow=num)
        return outputss


def resize_img(img, refsize):
    return F.pad(img, (2, 2, 2, 2)).expand(img.size(0), *refsize)
