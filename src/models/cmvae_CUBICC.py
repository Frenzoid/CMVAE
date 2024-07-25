# CUBICC CMVAE model specification
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from numpy import sqrt, prod
import torch.distributions as dist
from torchvision.utils import save_image, make_grid
from utils import Constants, plot_text_as_image_tensor
from .cmvae import CMVAE
from .vae_CUBICC_image_modality import CUB_Image
from .vae_CUBICC_captions_modality import CUB_Sentence


# Constants
maxSentLen = 32
minOccur = 3


class CUB_Image_Sentence(CMVAE):

    def __init__(self, params):
        super(CUB_Image_Sentence, self).__init__(dist.Normal if params.priorposterior == 'Normal' else dist.Laplace, params, CUB_Image, CUB_Sentence)
        grad = {'requires_grad': False}
        grad_c = {'requires_grad': params.learn_prior_c}
        self._pc_params = nn.ParameterList([
            nn.Parameter(torch.zeros(1, params.latent_dim_c), **grad_c)
        ])
        self._pz_params_m = nn.ParameterList([
            nn.Parameter(((2*torch.rand(1, params.latent_dim_z))-1)/2, requires_grad=True) for c_k in range(params.latent_dim_c)
        ])
        self._pz_params_lv = nn.ParameterList([
            nn.Parameter(torch.zeros(1, params.latent_dim_z), **grad) for c_k in range(params.latent_dim_c)
        ])
        self._pw_params = nn.ParameterList([
            nn.Parameter(torch.zeros(1, params.latent_dim_w), requires_grad=False),  # mu
            nn.Parameter(torch.zeros(1, params.latent_dim_w), requires_grad=False)  # logvar
        ])
        self.vaes[0].llik_scaling = self.vaes[1].maxSentLen / prod(self.vaes[0].dataSize) #\
        #     if params.llik_scaling == 0 else params.llik_scaling
        self.vaes[1].llik_scaling = params.llik_scaling_sent
        self.modelName = 'CMVAE_CUBICC'
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
            pc_a[:, idx_to_anneal] = -1e30
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
            recons_mat = super(CUB_Image_Sentence, self).self_and_cross_modal_generation([d[:num] for d in data])
            for r, recons_list in enumerate(recons_mat):
                for o, recon in enumerate(recons_list):
                    if o == 0:
                        recon = recon.squeeze(0).cpu()
                        recon_triess[r][o].append(recon)
                    else:
                        if i < 3: # Lower number of generations for cross-modal text generation
                            recon = recon.squeeze(0).cpu()
                            recon_triess[r][o].append(self._plot_sentences_as_tensor(recon))
        for r, recons_list in enumerate(recons_mat):
            if r == 1:
                input_data = self._plot_sentences_as_tensor(data[r][:num]).cpu()
            else:
                input_data = data[r][:num].cpu()

            for o, recon in enumerate(recons_list):
                outputss[r][o] = make_grid(torch.cat([input_data]+recon_triess[r][o], dim=2), nrow=num)
        return outputss

    def self_and_cross_modal_generation_for_fid_calculation(self, data, savePath, i):
        """
        Self- and cross-modal reconstruction for FID calculation.
        Args:
            data: Input
            savePath: Save path
            i: image index for naming
        """
        recons_mat = super(CUB_Image_Sentence, self).self_and_cross_modal_generation([d for d in data])
        for r, recons_list in enumerate(recons_mat):
            for o, recon in enumerate(recons_list):
                if o == 0:
                    recon = recon.squeeze(0).cpu()

                    for image in range(recon.size(0)):
                        save_image(recon[image, :, :, :],
                                    '{}/m{}/m{}/{}_{}.png'.format(savePath, r,o, image, i))

    def save_test_samples_for_fid_calculation(self, data, savePath, i):
        """
        Save test samples for FID calculation.
        Args:
            data: Input
            savePath: Save path
            i: image index for naming
        """
        o = 0
        imgs = data[0].cpu()
        for image in range(imgs.size(0)):
            save_image(imgs[image, :, :, :], '{}/test/m{}/{}_{}.png'.format(savePath, o, image, i))

    def _plot_sentences_as_tensor(self, batched_text_modality):
        """

        Args:
            batched_text_modality: Gets batch of text modality (as it is in input to forward function)

        Returns:
            Tensor with a corresponding plot containing the given text
        """
        sentences_processed = self._sent_process(batched_text_modality.argmax(-1))
        sentences_worded = [' '.join(self.vaes[1].i2w[str(word)] for word in sent if self.vaes[1].i2w[str(word)] != '<pad>') for sent in sentences_processed]
        return plot_text_as_image_tensor(sentences_worded, pixel_width=64, pixel_height=384)

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
            samples_list = super(CUB_Image_Sentence, self).generate_random_unconditional(N)
        if indexes_to_prune is None and indexes_to_select is None and random == False:
            samples_list = super(CUB_Image_Sentence, self).generate_unconditional(N)
        if indexes_to_prune is not None and indexes_to_select is None and random == True:
            samples_list = super(CUB_Image_Sentence, self).generate_random_unconditional_with_pruning(N, indexes_to_prune)
        if indexes_to_prune is None and indexes_to_select is not None and random == False:
            samples_list = super(CUB_Image_Sentence, self).generate_unconditional_with_input_latent_clusters(int(sqrt(N)), indexes_to_select)
        if coherence_calculation:
            return [samples.data.cpu() for samples in samples_list]
        elif fid_calculation:
            for i, samples in enumerate(samples_list):
                if i == 0:
                    # Only first modality (images) is considered for FID calculation
                    samples = samples.data.cpu()
                    # wrangle things so they come out tiled
                    # samples = samples.view(N, *samples.size()[1:])
                    for image in range(samples.size(0)):
                        save_image(samples[image, :, :, :], '{}/random/m{}/{}_{}.png'.format(savePath, i, tranche, image))
                else:
                    continue
        else:
            for i, samples in enumerate(samples_list):
                if i == 0:
                    # Image modality
                    samples = samples.data.cpu()
                    samples = samples.view(samples.size()[0], *samples.size()[1:])
                    outputs.append(make_grid(samples, nrow=int(sqrt(N))))
                else:
                    # Text modality
                    samples = samples.data.cpu()
                    samples = samples.view(samples.size()[0], *samples.size()[1:])
                    outputs.append(make_grid(self._plot_sentences_as_tensor(samples), nrow=int(sqrt(N))))

        return outputs

    def _sent_process(self, sentences):
        return [self.vaes[1].fn_trun(self.vaes[1].fn_2i(s)) for s in sentences]


