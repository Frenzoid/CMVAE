# objectives of choice
import torch
from numpy import prod
from utils import log_mean_exp, is_multidata
import torch.nn.functional as F

eta = 1e-20

# helper to vectorise computation
def compute_microbatch_split(x, K):
    """ Checks if batch needs to be broken down further to fit in memory. """
    B = x[0].size(0) if is_multidata(x) else x.size(0)
    S = sum([1.0 / (K * prod(_x.size()[1:])) for _x in x]) if is_multidata(x) \
        else 1.0 / (K * prod(x.size()[1:]))
    S = int(1e8 * S)  # float heuristic for 12Gb cuda memory
    assert (S > 0), "Cannot fit individual data in memory, consider smaller K"
    return min(B, S)


# MULTIMODAL OBJECTIVES

def _cmvae_iwae(model, x, K=1):
    """IWAE estimate for log p_\theta(x) for multi-modal vae -- fully vectorised
    This version is the looser bound---with the average over modalities outside the log
    """
    qu_xs, px_us, uss = model(x, K)
    qz_xs, qw_xs = [], []
    for r, qu_x in enumerate(qu_xs):
        qu_x_r_mean, qu_x_r_lv = model.vaes[r].qu_x_params
        qw_x_mean, qz_x_mean = torch.split(qu_x_r_mean, [model.params.latent_dim_w, model.params.latent_dim_z], dim=-1)
        qw_x_lv, qz_x_lv = torch.split(qu_x_r_lv, [model.params.latent_dim_w, model.params.latent_dim_z], dim=-1)
        qw_x = model.vaes[r].qu_x(qw_x_mean, qw_x_lv)
        qz_x = model.vaes[r].qu_x(qz_x_mean, qz_x_lv)
        qz_xs.append(qz_x)
        qw_xs.append(qw_x)
    lws = []
    for r, qz_x in enumerate(qz_xs):
        ws, zs = torch.split(uss[r], [model.params.latent_dim_w, model.params.latent_dim_z], dim=-1)

        pc = model.pc_params
        lpc = torch.log(pc)
        lpc = lpc.unsqueeze(1).repeat((1, zs.size()[1], 1))
        lpz_c_l = [model.pz(*model.pz_params(idx)).log_prob(zs).sum(-1) for idx in
                   range(model.params.latent_dim_c)]
        lpz_c = torch.stack(lpz_c_l, dim=-1)
        pc_z = F.softmax((lpc + lpz_c), dim=-1) + eta
        lpc_z = torch.log(pc_z)
        lpw = model.pw(*model.pw_params).log_prob(ws).sum(-1)

        lqz_x = log_mean_exp(torch.stack([qz_x.log_prob(zs).sum(-1) for qz_x in qz_xs]))
        lqw_x = qw_xs[r].log_prob(ws).sum(-1)
        lpx_u = [px_u.log_prob(x[d]).view(*px_u.batch_shape[:2], -1)
                     .mul(model.vaes[d].llik_scaling).sum(-1)
                 for d, px_u in enumerate(px_us[r])]

        lpx_u = torch.stack(lpx_u).sum(0)
        lw = lpx_u + model.params.beta * (((lpz_c + lpc - lpc_z) * pc_z).sum(-1) - lqz_x - lqw_x + lpw) #- (gamma_ada * ent_pc)
        lws.append(lw)
    return torch.stack(lws)  # (n_modality * n_samples) x batch_size, batch_size

def cmvae_iwae(model, x, K=1):
    """Computes iwae estimate for log p_\theta(x) for multi-modal vae
    This version is the looser bound---with the average over modalities outside the log
    """
    S = compute_microbatch_split(x, K)
    x_split = zip(*[_x.split(S) for _x in x])
    lw = [_cmvae_iwae(model, _x, K) for _x in x_split]
    lw = torch.cat(lw, 2)  # concat on batch
    return log_mean_exp(lw, dim=1).mean(0).sum()


def _cmvae_dreg(model, x, K=1):
    qu_xs, px_us, uss = model(x, K)
    qu_xs_ = [vae.qu_x(*[p.detach() for p in vae.qu_x_params]) for vae in model.vaes]
    qz_xs, qw_xs = [], []
    for r, qu_x in enumerate(qu_xs_):
        qu_x_r_mean, qu_x_r_lv = model.vaes[r].qu_x_params
        qw_x_mean, qz_x_mean = torch.split(qu_x_r_mean, [model.params.latent_dim_w, model.params.latent_dim_z], dim=-1)
        qw_x_lv, qz_x_lv = torch.split(qu_x_r_lv, [model.params.latent_dim_w, model.params.latent_dim_z], dim=-1)
        qw_x = model.vaes[r].qu_x(qw_x_mean, qw_x_lv)
        qz_x = model.vaes[r].qu_x(qz_x_mean, qz_x_lv)
        qz_xs.append(qz_x)
        qw_xs.append(qw_x)
    lws = []
    for r, qz_x in enumerate(qz_xs):
        ws, zs = torch.split(uss[r], [model.params.latent_dim_w, model.params.latent_dim_z], dim=-1)

        pc = model.pc_params
        lpc = torch.log(pc)
        lpc = lpc.unsqueeze(1).repeat((1, zs.size()[1], 1))
        lpz_c_l = [model.pz(*model.pz_params(idx)).log_prob(zs).sum(-1) for idx in
                   range(model.params.latent_dim_c)]
        lpz_c = torch.stack(lpz_c_l, dim=-1)
        pc_z = F.softmax((lpc + lpz_c), dim=-1) + eta
        lpc_z = torch.log(pc_z)
        lpw = model.pw(*model.pw_params).log_prob(ws).sum(-1)

        lqz_x = log_mean_exp(torch.stack([qz_x.log_prob(zs).sum(-1) for qz_x in qz_xs]))
        lqw_x = qw_xs[r].log_prob(ws).sum(-1)
        lpx_u = [px_u.log_prob(x[d]).view(*px_u.batch_shape[:2], -1)
                     .mul(model.vaes[d].llik_scaling).sum(-1)
                 for d, px_u in enumerate(px_us[r])]

        lpx_u = torch.stack(lpx_u).sum(0)
        lw = lpx_u + model.params.beta * (((lpz_c + lpc - lpc_z) * pc_z).sum(-1) - lqz_x - lqw_x + lpw)
        lws.append(lw)

    return torch.stack(lws), torch.stack(uss)

def cmvae_dreg(model, x, K=1):
    """Computes dreg estimate for log p_\theta(x) for multi-modal vae
    This version is the looser bound---with the average over modalities outside the log
    """
    S = compute_microbatch_split(x, K)
    x_split = zip(*[_x.split(S) for _x in x])
    lw, uss = zip(*[_cmvae_dreg(model, _x, K) for _x in x_split])
    lw = torch.cat(lw, 2)  # concat on batch
    uss = torch.cat(uss, 2)  # concat on batch
    with torch.no_grad():
        grad_wt = (lw - torch.logsumexp(lw, 1, keepdim=True)).exp()
        if uss.requires_grad:
            uss.register_hook(lambda grad: grad_wt.unsqueeze(-1) * grad)
    return (grad_wt * lw).mean(0).sum()


