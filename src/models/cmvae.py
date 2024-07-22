# Base CMVAE class definition
import torch
import torch.nn as nn
from utils import get_mean
import torch.distributions as dist

class CMVAE(nn.Module):
    """
    CMVAE class definition. Multimodal VAE with clustering in the latent space.
    """
    def __init__(self, prior_dist, params, *vaes):
        super(CMVAE, self).__init__()
        self.pz = prior_dist # Prior distribution (shared latent)
        self.pw = prior_dist # Prior distribution (modality-specific latent)
        self.vaes = nn.ModuleList([vae(params) for vae in vaes]) # List of unimodal VAEs (one for each modality)
        self.modelName = None  # Filled-in in subclass
        self.params = params # Model parameters (i.e. args passed to main script)

    @staticmethod
    def getDataSets(batch_size, shuffle=True, device="cuda"):
        # Handle getting individual datasets appropriately in sub-class
        raise NotImplementedError

    @property
    def pw_params(self):
        """Handled in multimodal VAE subclass, depends on the distribution class"""
        return self._pw_params

    def forward(self, x, K=1):
        """
        Forward function.
        Input:
            - x: list of data samples for each modality
            - K: number of samples for reparameterization in latent space

        Returns:
            - qu_xs: List of encoding distributions (one per encoder)
            - px_us: Matrix of self- and cross- reconstructions. px_us[m][n] contains
                    m --> n  reconstruction.
            - uss: List of latent codes, one for each modality. uss[m] contains latents inferred
                   from modality m. Note there latents are the concatenation of private and shared latents.
        """
        qu_xs, uss = [], []
        px_us = [[None for _ in range(len(self.vaes))] for _ in range(len(self.vaes))]
        # Loop over unimodal vaes
        for m, vae in enumerate(self.vaes):
            qu_x, px_u, us = vae(x[m], K=K) # Get Encoding dist, Decoding dist, Latents for unimodal VAE m modality
            qu_xs.append(qu_x) # Append encoding distribution to list
            uss.append(us) # Append latents to list
            px_us[m][m] = px_u  # Fill-in self-reconstructions in the matrix
        # Loop over unimodal vaes and compute cross-modal reconstructions
        for e, us in enumerate(uss):
            for d, vae in enumerate(self.vaes):
                if e != d:  # fill-in off-diagonal with cross-modal reconstructions
                    # Get shared latents from encoding modality e
                    _, z_e = torch.split(us, [self.params.latent_dim_w, self.params.latent_dim_z], dim=-1)
                    # Resample modality-specific encoding from modality-specific auxiliary distribution for decoding modality m
                    pw = vae.pw(*vae.pw_params_aux)
                    latents_w = pw.rsample(torch.Size([us.size()[0], us.size()[1]])).squeeze(2)
                    # Fixed for cuda (sorry)
                    if not self.params.no_cuda and torch.cuda.is_available():
                        latents_w.cuda()
                    # Combine shared and resampled private latents
                    us_combined = torch.cat((latents_w, z_e), dim=-1)
                    # Get cross-reconstruction likelihood
                    px_us[e][d] = vae.px_u(*vae.dec(us_combined))
        return qu_xs, px_us, uss


    def generate_random_unconditional(self, N):
        """
        Unconditional random generation.
        Args:
            N: Number of samples to generate.
        Returns:
            Generations
        """
        with torch.no_grad():
            data = []
            # Sample N latent clusters
            idxs = dist.Categorical(probs=self.pc_params).sample([N])
            latents_z_l = []
            for idx in idxs:
                pz = self.pz(*self.pz_params(idx))
                latents_z = pz.rsample(torch.Size([1]))
                latents_z_l.append(latents_z)
            latents_z_all = torch.cat(latents_z_l, dim=0)
            # Decode for all modalities
            for d, vae in enumerate(self.vaes):
                pw = self.pw(*self.pw_params)
                latents_w = pw.rsample([latents_z_all.size()[0]])
                latents = torch.cat((latents_w, latents_z_all), dim=-1)
                px_u = vae.px_u(*vae.dec(latents))
                data.append(px_u.mean.view(-1, *px_u.mean.size()[2:]))
        return data  # list of generations---one for each modality
    

    def generate_random_unconditional_with_pruning(self, N, idxs_to_prune):
        """
        Unconditional random generation with pruned clusters.
        Args:
            N: Number of samples to generate
            idxs_to_anneal: Indexes of annealed latent clusters

        Returns:
            Generations
        """
        with torch.no_grad():
            data = []
            if idxs_to_prune ==  None:
                idxs = dist.Categorical(probs=self.pc_params).sample([N])
            else: 
                idxs = dist.Categorical(probs=self.pc_params_pruning(idxs_to_prune)).sample([N])
            latents_z_l = []
            for idx in idxs:
                pz = self.pz(*self.pz_params(idx))
                latents_z = pz.rsample(torch.Size([1]))
                latents_z_l.append(latents_z)
            latents_z_all = torch.cat(latents_z_l, dim=0)
            # Decode for all modalities
            for d, vae in enumerate(self.vaes):
                pw = self.pw(*self.pw_params)
                latents_w = pw.rsample([latents_z_all.size()[0]])
                latents = torch.cat((latents_w, latents_z_all), dim=-1)
                px_u = vae.px_u(*vae.dec(latents))
                data.append(px_u.mean.view(-1, *px_u.mean.size()[2:]))
        return data  # list of generations---one for each modality


    def generate_unconditional(self, N):
        """
        Unconditional generation from each latent cluster.
        Args:
            N: Number of samples to generate

        Returns:
            Generations
        """
        with torch.no_grad():
            data = []
            latents_z_l = []
            for idx in range(self.params.latent_dim_c):
                pz = self.pz(*self.pz_params(idx))
                latents_z = pz.rsample(torch.Size([N]))
                latents_z_l.append(latents_z)
            latents_z_all = torch.cat(latents_z_l, dim=0)
            # Decode for all modalities
            for d, vae in enumerate(self.vaes):
                pw = self.pw(*self.pw_params)
                latents_w = pw.rsample([latents_z_all.size()[0]])
                latents = torch.cat((latents_w, latents_z_all), dim=-1)
                px_u = vae.px_u(*vae.dec(latents))
                data.append(px_u.mean.view(-1, *px_u.mean.size()[2:]))
        return data  # list of generations---one for each modality

    def generate_unconditional_with_input_latent_clusters(self, N, indexes):
        """
        Unconditional generation from selected latent clusters.
        Args:
            indexes: list of latent clusters indexes to sample from

        Returns:
            Generations
        """
        with torch.no_grad():
            data = []
            latents_z_l = []
            for idx in indexes:
                pz = self.pz(*self.pz_params(idx))
                latents_z = pz.rsample(torch.Size([N]))
                latents_z_l.append(latents_z)
            latents_z_all = torch.cat(latents_z_l, dim=0)
            # Decode for all modalities
            for d, vae in enumerate(self.vaes):
                pw = self.pw(*self.pw_params)
                latents_w = pw.rsample([latents_z_all.size()[0]])
                latents = torch.cat((latents_w, latents_z_all), dim=-1)
                px_u = vae.px_u(*vae.dec(latents))
                data.append(px_u.mean.view(-1, *px_u.mean.size()[2:]))
        return data  # list of generations---one for each modality



    def self_and_cross_modal_generation_forward(self, x, K=1):
        """
        Test-time self- and cross-modal generation forward function.
        Input:
            - x: list of data samples for each modality
            - K: number of samples for reparameterization in latent space

        Returns:
            - qu_xs: List of encoding distributions (one per encoder)
            - px_us: Matrix of test-time self- and cross- reconstructions. px_us[m][n] contains
                    m --> n  reconstruction.
            - uss: List of latent codes, one for each modality. uss[m] contains latents inferred
                   from modality m. Note there latents are the concatenation of private and shared latents.
        """
        qu_xs, uss = [], []
        px_us = [[None for _ in range(len(self.vaes))] for _ in range(len(self.vaes))]
        # Loop over unimodal vaes
        for m, vae in enumerate(self.vaes):
            qu_x, px_u, us = vae(x[m], K=K) # Get Encoding dist, Decoding dist, Latents for unimodal VAE m modality
            qu_xs.append(qu_x) # Append encoding distribution to list
            uss.append(us) # Append latents to list
            px_us[m][m] = px_u  # Fill-in self-reconstructions in the matrix
        # Loop over unimodal vaes and compute cross-modal reconstructions
        for e, us in enumerate(uss):
            for d, vae in enumerate(self.vaes):
                if e != d:  # fill-in off-diagonal with cross-modal reconstructions
                    # Get shared latents from encoding modality e
                    _, z_e = torch.split(us, [self.params.latent_dim_w, self.params.latent_dim_z], dim=-1)
                    # Resample modality-specific encoding from modality-specific auxiliary distribution for decoding modality m
                    pw = vae.pw(*vae.pw_params_std)
                    latents_w = pw.rsample(torch.Size([us.size()[0], us.size()[1]])).squeeze(2)
                    # Fixed for cuda (sorry)
                    if not self.params.no_cuda and torch.cuda.is_available():
                        latents_w.cuda()
                    # Combine shared and resampled private latents
                    us_combined = torch.cat((latents_w, z_e), dim=-1)
                    # Get cross-reconstruction likelihood
                    px_us[e][d] = vae.px_u(*vae.dec(us_combined))
        return qu_xs, px_us, uss

    def self_and_cross_modal_generation(self, data):
        """
        Test-time self- and cross-reconstruction.
        Args:
            data: Input

        Returns:
            Matrix of self- and cross-modal reconstructions

        """
        with torch.no_grad():
            _, px_us, _ = self.self_and_cross_modal_generation_forward(data)
            # ------------------------------------------------
            # cross-modal matrix of reconstructions
            recons = [[get_mean(px_u) for px_u in r] for r in px_us]
        return recons
