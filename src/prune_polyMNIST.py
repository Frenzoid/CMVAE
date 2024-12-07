"""
Pruning procedure on PolyMNIST dataset
"""
import os
import sys
import argparse
import numpy as np
import torch
import torch.nn.functional as F
# Relative import hacks
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)  # for bash user
os.chdir(parentdir)  # for pycharm user
import models
from utils import unpack_data
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from statistics import mean
from scipy.stats import entropy
from torch.utils.data import Subset
import wandb
from utils import Constants
from utils import cluster_acc

# torch.backends.cudnn.benchmark = True

# Parsing commands
parser = argparse.ArgumentParser(description='Pruning')
parser.add_argument('--save-dir', type=str, default="../outputs/PolyMNIST_1/checkpoints/32_32_2.5_2",
                    metavar='N', help='directory where model is saved')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA use')
parser.add_argument('--epoch', type=int, default=250,
                    help='epoch from which we load the model')
parser.add_argument('--seed', type=int, default=2,
                    help='random seed')
parser.add_argument('--datadir', type=str, default="../data", help="directory where data is saved")


# Parse commands
cmds = parser.parse_args()

if not os.path.exists(cmds.save_dir):
    runpath_temp = cmds.save_dir.removeprefix(".")
    if not os.path.exists(runpath_temp):
        print("Couldn't find the path to outputs directory.")
        raise FileNotFoundError
    else:
        cmds.save_dir = runpath_temp
        
if not os.path.exists(cmds.datadir):
    datadir_path = cmds.datadir.removeprefix(".")
    if not os.path.exists(datadir_path):
        print("Couldn't find the path to data directory.")
        raise FileNotFoundError
    else:
        cmds.datadir = datadir_path

runPath = cmds.save_dir
print(runPath)

# Set seed
torch.manual_seed(cmds.seed)

# Parse args from trained model
args = torch.load(os.path.join(runPath, 'args.rar'))

# Set datadir to current datadir (to match the path where data can be found)
args.datadir = cmds.datadir

# WandB
wandb.login()

wandb.init(
    # Set the project where this run will be logged
    project=args.experiment,
    # Track hyperparameters and run metadata
    config=args,
    # Run name
    name= str(args.latent_dim_w) + '_' + str(args.latent_dim_z) + '_' + str(args.beta) + '_' + str(args.seed) + "_pruning"
)

# CUDA stuff
args.cuda = not cmds.no_cuda and torch.cuda.is_available()
needs_conversion = not args.cuda
conversion_kwargs = {'map_location': lambda st, loc: st} if needs_conversion else {}
device = torch.device("cuda" if args.cuda else "cpu")

modelC = getattr(models, 'CMVAE_PolyMNIST_5modalities')
model = modelC(args)
if args.cuda:
    model.cuda()

# Load model
model.load_state_dict(torch.load(runPath + '/model_{}.rar'.format(cmds.epoch), **conversion_kwargs), strict=False)
# Get train and test loader (note the test set is further split into validation and test)
train_dataset, test_and_validation_dataset = model.getDataSets(args.batch_size, device=device)
# Load validation and test indices
validation_indices = np.load(os.path.join(args.datadir, 'valid_PMtest_indices.npy'))
test_indices = np.load(os.path.join(args.datadir, 'test_PMtest_indices.npy'))
validation_dataset = Subset(test_and_validation_dataset, validation_indices)
test_dataset = Subset(test_and_validation_dataset, test_indices)
kwargs = {'num_workers': 2, 'pin_memory': True} if device == 'cuda' else {}

# Set validation and test loaders
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, **kwargs)
validation_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=args.batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, **kwargs)


# Single step of pruning clusters procedure
def prune_clusters_and_calculate_metrics_single_step(dl, idxs_to_prune=None):
    """
    Inputs:
        dl: dataloader to use
        idxs_to_prune: list of clusters to prune. None for no clusters pruned
    Outputs:
        nmi over dataset,
        ari over dataset,
        acc over dataset,
        generations for active (i.e. non-pruned) clusters,
        dictionary of probability mass per cluster,
        penalized normalized entropy with current latent clusters pruned
    """
    with torch.no_grad():
        #  Create lists to store labels and predicted labels when iterating over the dataloader
        labels, cluster_assignments = [], []
        # Create list to contain entropy and logliks values
        entropies, lliks = [], []
        # Creating dict to store probality mass per cluster
        prob_mass_per_cluster = {val: 0 for val in range(model.params.latent_dim_c)}
        # Itearate over dataloader
        for i, dataT in enumerate(dl):
            # Unpack data
            data, labels_b = unpack_data(dataT, device=device)
            # Inference
            qu_xs, _, uss = model(data, K=1)
            # Helper lists
            cluster_assignments_single_datapoint = []
            norm_entropies_single_datapoint = []
            lliks_single_datapoint = []
            # Iterate over modalites for encoding (encoding distributions)
            pc_zs_single_datapoint = []
            for r, qu_x in enumerate(qu_xs):
                us = uss[r]
                # Split unimodal latent code zs in modality-specific (ws) and shared latent code (us)
                ws, zs = torch.split(us, [model.params.latent_dim_w, model.params.latent_dim_z], dim=-1)
                # Prune clusters if specified
                if idxs_to_prune is not None:
                    pc = model.pc_params_pruning(idxs_to_prune)
                else:
                    pc = model.pc_params
                # Calculate necessary components of the ELBO
                lpc = torch.log(pc + Constants.eta) # eta added for numerical stability
                lpc = lpc.unsqueeze(1).repeat((1, zs.size()[1], 1))
                lpz_c = torch.stack([model.pz(*model.pz_params(idx)).log_prob(zs).sum(-1) for idx in
                           range(model.params.latent_dim_c)], dim=-1)
                pc_z = F.softmax((lpc + lpz_c), dim=-1) + Constants.eta # eta added for numerical stability
                pc_zs_single_datapoint.append(pc_z)
                lpc_z = torch.log(pc_z)
                normalized_lliks = ((lpz_c + lpc - lpc_z) * pc_z).sum(-1).squeeze(0) / model.params.latent_dim_z # Divide by the number of dimensions of shared encoding
                lliks_single_datapoint.append(normalized_lliks)
                # Calculate normalized entropy
                ent_pc_z = torch.Tensor(entropy(pc_z.squeeze(0).cpu().numpy(), axis=1) / (np.log(np.count_nonzero(pc_z.squeeze(0).cpu().numpy(), axis=1)))).cuda()
                norm_entropies_single_datapoint.append(ent_pc_z)
                cluster_assignments_modality = pc_z.argmax(-1).squeeze(0)
                cluster_assignments_single_datapoint.append(cluster_assignments_modality)
            # Take the mode of cluster assignments across modalities as the final cluster assignment
            pc_zs = torch.stack(pc_zs_single_datapoint, dim=-1).squeeze(0)
            cluster_assignments_agg = torch.mode(torch.stack(cluster_assignments_single_datapoint, dim=1), dim=1)[0]
            # _, cluster_assignments_agg, pc_zs_agg = get_cluster_assignments_aggregate_modalities(pc_zs)

            # Update probabiliy mass per cluster
            for val in range(model.params.latent_dim_c):
                prob_mass_per_cluster[val] = prob_mass_per_cluster[val] + ((cluster_assignments_agg == val) * pc_zs.mean(-1).max(-1)[0]).sum().item()

            # Average log-likelihood across modalities
            lliks_single_datapoint_avg = torch.stack(lliks_single_datapoint, dim=0).mean(0)
            # Average penalilized norm entropy across modalities
            norm_entropies_single_datapoint_avg = torch.stack(norm_entropies_single_datapoint, dim=0).mean(0)

            # Save lliks, entropies, labels and cluster assignments for this batch
            lliks.append((lliks_single_datapoint_avg.mean()).item())
            entropies.append((norm_entropies_single_datapoint_avg.mean()).item())
            labels.append(labels_b.cpu())
            cluster_assignments.append(cluster_assignments_agg.cpu())


        # Compute penalized norm entropy
        penalized_norm_entropy =  (model.params.beta * mean(entropies)) - mean(lliks)
        prob_mass_per_cluster_active_only =  {k: v for k, v in prob_mass_per_cluster.items() if not v == 0} # Get probability mass only for active clusters
        # Generate from active clusters only
        generations_active_clusters = model.generate_unconditional(N=100, indexes_to_select=sorted(prob_mass_per_cluster_active_only, key=prob_mass_per_cluster_active_only.get, reverse=True), random=False)
        labels_toevalmetrics = torch.cat(labels, dim=-1) # Concat labels for NMI and ARI
        cluass_toevalmetrics = torch.cat(cluster_assignments, dim=-1) # Concat cluster assignments for NMI and ARI
        nmi = normalized_mutual_info_score(labels_toevalmetrics, cluass_toevalmetrics)
        ari =  adjusted_rand_score(labels_toevalmetrics, cluass_toevalmetrics)
        acc = cluster_acc(labels_toevalmetrics.cpu().numpy(), cluass_toevalmetrics.cpu().numpy())
        print("NMI: ", nmi)
        print("ARI: ", ari)
        print("ACC: ", acc)
        print("N_pruned: ", len(idxs_to_prune) if idxs_to_prune is not None else None)
        print("PNE: ", penalized_norm_entropy)
    return nmi, ari, acc, generations_active_clusters, prob_mass_per_cluster, penalized_norm_entropy

def prune_clusters_and_calculate_metrics():
    with torch.no_grad():
        pruned = []
        # Start procedure (no pruning)
        nmi, ari, acc, generations_active_clusters, prob_mass_per_cluster, penalized_norm_entropy = prune_clusters_and_calculate_metrics_single_step(validation_loader, idxs_to_prune=None)
        # Initialize the min value as the value without pruning
        min_value_penalized_norm_entropy = penalized_norm_entropy
        clusters_to_prune_for_min_value_pne = None
        # Save generations for each active (i.e. non-pruned) clusters
        to_log_wandb = {'Generations/Active_Clusters/m{}'.format(idx_topk): wandb.Image(generations_active_clusters[idx_topk]) for idx_topk in range(len(model.vaes))}
        to_log_wandb['Metrics/Validation/NMI'] = nmi
        to_log_wandb['Metrics/Validation/ARI'] = ari
        to_log_wandb['Metrics/Validation/Acc'] = acc
        to_log_wandb['Metrics/Validation/Penalized_Norm_Entropy'] = penalized_norm_entropy
        wandb.log(to_log_wandb, step=0, commit=True)

        # Initially the dictionary with probability mass per cluster after pruning as the one without pruning
        prob_mass_per_cluster_after_pruning = {key: prob_mass_per_cluster[key] for key in prob_mass_per_cluster.keys()}
        # Iteratively prune clusters
        for c, _ in enumerate(sorted(prob_mass_per_cluster, key=prob_mass_per_cluster.get)):
            if c == (len(sorted(prob_mass_per_cluster, key=prob_mass_per_cluster.get)) - 1): # Stop at the end of the procedure
                 break
            # Filter only active (i.e. non-pruned) clusters
            prob_mass_per_active_cluster = {key: prob_mass_per_cluster_after_pruning[key] for key in prob_mass_per_cluster_after_pruning.keys() if key not in pruned}

            # Get index of cluster to prune at this iteration (lowest probability mass among active clusters)
            idx_to_prune = sorted(prob_mass_per_active_cluster, key=prob_mass_per_active_cluster.get, reverse=True)[-1]
            pruned.append(idx_to_prune) # Append cluster to pruned clusters list
            # Repeat clustering in the latent space with pruned clusters updated
            nmi, ari, acc, generations_active_clusters, prob_mass_per_cluster_after_pruning, penalized_norm_entropy = prune_clusters_and_calculate_metrics_single_step(validation_loader, idxs_to_prune=pruned)

            # Check if penalized norm entropy is less than the minimum achieved so far
            if penalized_norm_entropy < min_value_penalized_norm_entropy and c != (len(sorted(prob_mass_per_cluster, key=prob_mass_per_cluster.get)) - 2): # Second condition is to avoid incurring in the degenerate case where there is only one cluster and entropy is trivially zero
                clusters_to_prune_for_min_value_pne = list(pruned)
                min_value_penalized_norm_entropy = penalized_norm_entropy

            to_log_wandb = {'Generations/Active_Clusters/m{}'.format(idx_topk): wandb.Image(generations_active_clusters[idx_topk]) for idx_topk in
                            range(len(model.vaes))}
            to_log_wandb['Metrics/Validation/NMI'] = nmi
            to_log_wandb['Metrics/Validation/ARI'] = ari
            to_log_wandb['Metrics/Validation/Acc'] = acc
            to_log_wandb['Metrics/Validation/Penalized_Norm_Entropy'] = penalized_norm_entropy
            wandb.log(to_log_wandb, step=c+1, commit=True)

        # Test on test dataset
        nmi_test, ari_test, acc_test, generations_active_clusters_test,_, _ = prune_clusters_and_calculate_metrics_single_step(test_loader, idxs_to_prune=clusters_to_prune_for_min_value_pne)
        to_log_wandb = {'Generations/Active_Clusters_Found_After_Pruning/m{}'.format(idx_topk): wandb.Image(generations_active_clusters_test[idx_topk]) for idx_topk in
                         range(len(model.vaes))}
        to_log_wandb['Metrics/Test/NMI'] = nmi_test
        to_log_wandb['Metrics/Test/ARI'] = ari_test
        to_log_wandb['Metrics/Test/Acc'] = acc_test
        to_log_wandb['Num_Clusters_Found_After_Pruning'] = model.params.latent_dim_c - len(clusters_to_prune_for_min_value_pne)
        wandb.log(to_log_wandb, commit=True)
    return pruned


if __name__ == '__main__':
    model.eval()
    with torch.no_grad():
        results = prune_clusters_and_calculate_metrics()
        print("Pruned clusters: ", results)


