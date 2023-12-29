from typing import Any

import torch
from pytorch_lightning.core.mixins.hparams_mixin import HyperparametersMixin
from torch import Tensor as T
from torch.nn.functional import mse_loss
from torch_cluster import radius_graph

from gnn_tracking.metrics.losses import MultiLossFct, MultiLossFctReturn
from gnn_tracking.utils.graph_masks import get_good_node_mask_tensors
from gnn_tracking.utils.log import logger

# ruff: noqa: ARG002


@torch.compile
def _first_occurrences(x: T) -> T:
    """Return the first occurrence of each unique element in a 1D array"""
    # from https://discuss.pytorch.org/t/first-occurrence-of-unique-values-in-a-tensor/81100/3
    unique, inverse = torch.unique(x, sorted=True, return_inverse=True)
    perm = torch.arange(inverse.size(0), dtype=inverse.dtype, device=inverse.device)
    inverse, perm = inverse.flip([0]), perm.flip([0])
    return inverse.new_empty(unique.size(0)).scatter_(0, inverse, perm)


@torch.compile
def _square_distances(edges: T, positions: T) -> T:
    """Returns squared distances between two sets of points"""
    return torch.sum((positions[edges[0]] - positions[edges[1]]) ** 2, dim=-1)


@torch.compile
def _get_alphas_first_occurences(beta: T, particle_id: T, mask: T) -> tuple[T, T]:
    sorted_indices_j = torch.argsort(beta[mask], descending=True)
    pids_sorted = particle_id[mask][sorted_indices_j]
    # Index of condensation points in node array
    alphas_k_masked = sorted_indices_j[_first_occurrences(pids_sorted)]
    assert alphas_k_masked.size()[0] > 0, "No particles found, cannot evaluate loss"
    # Now make it refer back to the original indices
    alphas_k = torch.nonzero(mask).squeeze()[alphas_k_masked]
    # 1D array (n_nodes): 1 for CPs, 0 otherwise
    is_cp_j = torch.zeros_like(particle_id, dtype=torch.bool).scatter_(0, alphas_k, 1)
    return alphas_k, is_cp_j


@torch.compile
def _get_vr_rg(
    *,
    radius_edges: T,
    is_cp_j: T,
    particle_id: T,
    x: T,
    q_j: T,
    radius_threshold: float,
):
    # Protect against sqrt not being differentiable around 0
    eps = 1e-9
    # Now filter out everything that doesn't include a CP or connects two hits of the
    # same particle
    to_cp_e = is_cp_j[radius_edges[0]]
    is_repulsive_e = particle_id[radius_edges[0]] != particle_id[radius_edges[1]]
    # Since noise/low pt does not have CPs, they don't repel from each other
    repulsion_edges_e = radius_edges[:, is_repulsive_e & to_cp_e]
    repulsion_distances_e = radius_threshold - torch.sqrt(
        eps + _square_distances(repulsion_edges_e, x)
    )
    return torch.sum(
        repulsion_distances_e * q_j[repulsion_edges_e[0]] * q_j[repulsion_edges_e[1]]
    )


@torch.compile
def _get_va(*, alphas_k: T, is_cp_j: T, particle_id: T, x: T, q_j: T, mask: T) -> T:
    # hit-indices of all non-CPs
    non_cp_indices = torch.nonzero(~is_cp_j & mask).squeeze()
    # for each non-CP hit, the index of the corresponding CP
    corresponding_alpha = alphas_k[
        torch.searchsorted(particle_id[alphas_k], particle_id[non_cp_indices])
    ]
    attraction_edges_e = torch.stack((non_cp_indices, corresponding_alpha))
    attraction_distances_e = _square_distances(attraction_edges_e, x)
    return torch.sum(
        attraction_distances_e * q_j[attraction_edges_e[0]] * q_j[attraction_edges_e[1]]
    )


def _radius_graph_condensation_loss(
    *,
    beta: T,
    x: T,
    particle_id: T,
    q_min: float,
    mask: T,
    radius_threshold: float,
    max_num_neighbors: int,
) -> tuple[dict[str, T], dict[str, Any]]:
    """Extracted function for condensation loss. See `PotentialLoss` for details.

    Args:
        mask: Mask for objects cast to nodes
    """
    # For better readability, variables that are only relevant in one "block"
    # are prefixed with an underscore
    # _j means indexed as hits (... x n_hits)
    # _k means indexed as objects (... x n_objects_of_interest)
    # _e means indexed by edge
    # where n_objects_of_interest = len(unique(particle_id[mask]))

    alphas_k, is_cp_j = _get_alphas_first_occurences(
        beta=beta, particle_id=particle_id, mask=mask
    )

    q_j = torch.arctanh(beta) ** 2 + q_min

    _radius_edges = radius_graph(
        x=x, r=radius_threshold, max_num_neighbors=max_num_neighbors, loop=False
    )
    vr = _get_vr_rg(
        radius_edges=_radius_edges,
        is_cp_j=is_cp_j,
        particle_id=particle_id,
        x=x,
        q_j=q_j,
        radius_threshold=radius_threshold,
    )

    va = _get_va(
        alphas_k=alphas_k,
        is_cp_j=is_cp_j,
        particle_id=particle_id,
        x=x,
        q_j=q_j,
        mask=mask,
    )

    # -- 4. Simple postproc --

    if torch.isnan(vr).any():
        vr = torch.tensor([[0.0]])
        logger.warning("Repulsive loss is NaN")

    hit_is_noise_j = particle_id == 0

    n_hits = len(mask)
    # oi = of interest = not masked
    n_hits_oi = mask.sum()
    n_particles_oi = len(alphas_k)
    eps = 1e-9
    # every hit has a rep edge to every other CP except its own
    norm_rep = eps + (n_particles_oi - 1) * n_hits
    # need to subtract n_particle_oi to avoid double counting
    norm_att = eps + n_hits_oi - n_particles_oi

    losses = {
        "attractive": va / norm_att,
        "repulsive": vr / norm_rep,
        "coward": torch.mean(1 - beta[alphas_k]),
        "noise": torch.mean(beta[hit_is_noise_j]),
    }
    extra = {}
    return losses, extra


class CondensationLossRG(MultiLossFct, HyperparametersMixin):
    def __init__(
        self,
        *,
        lw_repulsive: float = 1.0,
        lw_noise: float = 0.0,
        lw_coward: float = 0.0,
        q_min: float = 0.01,
        pt_thld: float = 0.9,
        max_eta: float = 4.0,
        max_num_neighbors: int = 256,
        sample_pids: float = 1.0,
    ):
        """Implementation of condensation loss that uses radius graph instead
        calculating the whole n^2 distance matrix.

        Args:
            lw_repulsive: Loss weight for repulsive part of potential loss
            lw_noise: Loss weight for noise loss
            lw_background: Loss weight for background loss
            q_min (float, optional): See OC paper. Defaults to 0.01.
            pt_thld (float, optional): pt thld for interesting particles. Defaults to 0.9.
            max_eta (float, optional): eta thld for interesting particles. Defaults to 4.0.
            max_num_neighbors (int, optional): Maximum number of neighbors to consider
                for radius graphs. Defaults to 256.
            sample_pids (float, optional): Further subsample particles to conserve
                memory. Defaults to 1.0 (no sampling)
        """
        super().__init__()
        self.save_hyperparameters()

    def forward(
        self,
        *,
        beta: T,
        x: T,
        particle_id: T,
        reconstructable: T,
        pt: T,
        ec_hit_mask: T | None = None,
        eta: T,
        **kwargs,
    ) -> MultiLossFctReturn:
        if ec_hit_mask is not None:
            # If a post-EC node mask was applied in the model, then all model outputs
            # already include this mask, while everything gotten from the data
            # does not. Hence, we apply it here.
            particle_id = particle_id[ec_hit_mask]
            reconstructable = reconstructable[ec_hit_mask]
            pt = pt[ec_hit_mask]
        mask = get_good_node_mask_tensors(
            pt=pt,
            particle_id=particle_id,
            reconstructable=reconstructable,
            eta=eta,
            pt_thld=self.hparams.pt_thld,
            max_eta=self.hparams.max_eta,
        )
        if self.hparams.sample_pids < 1:
            sample_mask = (
                torch.rand_like(beta, dtype=torch.float16) < self.hparams.sample_pids
            )
            mask &= sample_mask
        # If there are no hits left after masking, then we get a NaN loss.
        assert mask.sum() > 0, "No hits left after masking"
        loss_dict, extra_dict = _radius_graph_condensation_loss(
            beta=beta,
            x=x,
            particle_id=particle_id,
            mask=mask,
            q_min=self.hparams.q_min,
            radius_threshold=1.0,
            max_num_neighbors=self.hparams.max_num_neighbors,
        )
        weight_dict = {
            "attractive": 1.0,
            "repulsive": self.hparams.lw_repulsive,
            "noise": self.hparams.lw_noise,
            "coward": self.hparams.lw_coward,
        }
        return MultiLossFctReturn(
            loss_dct=loss_dict,
            weight_dct=weight_dict,
            extra_metrics=extra_dict,
        )


@torch.compile
def condensation_loss_tiger(
    *,
    beta: T,
    x: T,
    object_id: T,
    object_mask: T,
    q_min: float,
    noise_threshold: int,
    max_n_rep: int,
) -> tuple[dict[str, T], dict[str, int | float]]:
    """Extracted function for torch compilation. See `condensation_loss_tiger` for
    docstring.

    Args:
        object_mask: Mask for the particles that should be considered for the loss
            this is broadcased to n_hits

    Returns:
        loss_dct: Dictionary of losses
        extra_dct: Dictionary of extra information
    """
    # _j means indexed by hits
    # _k means indexed by objects

    # To protect against nan in divisions
    eps = 1e-9

    unique_oids_k = torch.unique(object_id[object_mask])
    assert (
        len(unique_oids_k) > 0
    ), "No particles of interest found, cannot evaluate loss"
    # n_nodes x n_pids
    # The nodes in every column correspond to the hits of a single particle and
    # should attract each other
    attractive_mask_jk = object_id.view(-1, 1) == unique_oids_k.view(1, -1)

    q_j = torch.arctanh(beta) ** 2 + q_min
    assert not torch.isnan(q_j).any(), "q contains NaNs"

    # Index of condensation points in node array
    alphas_k = torch.argmax(q_j.view(-1, 1) * attractive_mask_jk, dim=0)

    # 1 x n_objs
    q_k = q_j[alphas_k].view(1, -1)
    qw_jk = q_j.view(-1, 1) * q_k

    # n_objs x n_outdim
    x_k = x[alphas_k]
    dist_jk = torch.cdist(x, x_k)

    # Calculate normalization factors
    # -------------------------------
    n_hits = len(object_mask)
    # oi = of interest = not masked
    n_hits_oi = object_mask.sum()
    n_particles_oi = len(alphas_k)
    # every hit has a rep edge to every other CP except its own
    norm_rep = eps + (n_particles_oi - 1) * n_hits
    # need to subtract n_particle_oi to avoid double counting
    norm_att = eps + n_hits_oi - n_particles_oi

    # Attractive potential/loss
    # -------------------------
    qw_att = qw_jk[attractive_mask_jk]
    v_att = (qw_att * torch.square(dist_jk[attractive_mask_jk])).sum() / norm_att

    # Repulsive potential
    # -------------------
    repulsive_mask_jk = (~attractive_mask_jk) & (dist_jk < 1)
    n_rep = repulsive_mask_jk.sum()
    if n_rep > max_n_rep > 0:
        sampling_freq = max_n_rep / n_rep
        sampling_mask = (
            torch.rand_like(repulsive_mask_jk, dtype=torch.float16) < sampling_freq
        )
        repulsive_mask_jk &= sampling_mask
        norm_rep *= sampling_freq
    qw_rep_jk = qw_jk[repulsive_mask_jk]
    v_rep = (qw_rep_jk * (1 - dist_jk[repulsive_mask_jk])).sum() / norm_rep

    # Other losses
    # ------------
    l_coward = torch.mean(1 - beta[alphas_k])
    not_noise_j = object_id > noise_threshold
    l_noise = torch.mean(beta[~not_noise_j])

    loss_dct = {
        "attractive": v_att,
        "repulsive": v_rep,
        "coward": l_coward,
        "noise": l_noise,
    }
    extra_dct = {
        "n_rep": n_rep,
    }
    return loss_dct, extra_dct


class CondensationLossTiger(MultiLossFct, HyperparametersMixin):
    def __init__(
        self,
        *,
        lw_repulsive: float = 1.0,
        lw_noise: float = 0.0,
        lw_coward: float = 0.0,
        q_min: float = 0.01,
        pt_thld: float = 0.9,
        max_eta: float = 4.0,
        max_n_rep: int = 0,
        sample_pids: float = 1.0,
    ):
        """Implementation of condensation loss that directly calculates the n^2
        distance matrix.

        Args:
            lw_repulsive: Loss weight for repulsive part of potential loss
            lw_noise: Loss weight for noise loss
            lw_background: Loss weight for background loss
            q_min (float, optional): See OC paper. Defaults to 0.01.
            pt_thld (float, optional): pt thld for interesting particles. Defaults to 0.9.
            max_eta (float, optional): eta thld for interesting particles. Defaults to 4.0.
            max_n_rep (int, optional): Maximum number of repulsive edges to consider.
                Defaults to 0 (all).
            sample_pids (float, optional): Further subsample particles to conserve
                memory. Defaults to 1.0 (no sampling)
        """
        super().__init__()
        self.save_hyperparameters()

    # noinspection PyUnusedLocal
    def forward(
        self,
        *,
        beta: T,
        x: T,
        particle_id: T,
        reconstructable: T,
        pt: T,
        ec_hit_mask: T | None = None,
        eta: T,
        **kwargs,
    ) -> MultiLossFctReturn:
        if ec_hit_mask is not None:
            # If a post-EC node mask was applied in the model, then all model outputs
            # already include this mask, while everything gotten from the data
            # does not. Hence, we apply it here.
            particle_id = particle_id[ec_hit_mask]
            reconstructable = reconstructable[ec_hit_mask]
            pt = pt[ec_hit_mask]
            eta = eta[ec_hit_mask]
        mask = get_good_node_mask_tensors(
            pt=pt,
            particle_id=particle_id,
            reconstructable=reconstructable,
            eta=eta,
            pt_thld=self.hparams.pt_thld,
            max_eta=self.hparams.max_eta,
        )
        if self.hparams.sample_pids < 1:
            sample_mask = (
                torch.rand_like(beta, dtype=torch.float16) < self.hparams.sample_pids
            )
            mask &= sample_mask
        # If there are no hits left after masking, then we get a NaN loss.
        assert mask.sum() > 0, "No hits left after masking"
        losses, extra = condensation_loss_tiger(
            beta=beta,
            x=x,
            object_id=particle_id,
            object_mask=mask,
            q_min=self.hparams.q_min,
            noise_threshold=0.0,
            max_n_rep=self.hparams.max_n_rep,
        )
        weights = {
            "attractive": 1.0,
            "repulsive": self.hparams.lw_repulsive,
            "noise": self.hparams.lw_noise,
            "coward": self.hparams.lw_coward,
        }
        return MultiLossFctReturn(
            loss_dct=losses,
            weight_dct=weights,
            extra_metrics=extra,
        )


class ObjectLoss(torch.nn.Module):
    def __init__(self, mode="efficiency"):
        """Loss functions for predicted object properties."""
        super().__init__()
        self.mode = mode

    @staticmethod
    def _mse(*, pred: T, truth: T) -> T:
        return torch.sum(mse_loss(pred, truth, reduction="none"), dim=1)

    def object_loss(self, *, pred: T, beta: T, truth: T, particle_id: T) -> T:
        # shape: n_nodes
        mse = self._mse(pred=pred, truth=truth)
        if self.mode == "purity":
            noise_mask = particle_id == 0
            # shape: n_nodes
            xi = (~noise_mask) * torch.arctanh(beta) ** 2
            return 1 / torch.sum(xi) * torch.mean(xi * mse)
        if self.mode == "efficiency":
            # shape: n_pids
            pids = torch.unique(particle_id[particle_id > 0])
            # PID masks (n_nodes x n_pids)
            pid_masks = particle_id[:, None] == pids[None, :]
            # shape: (n_nodes x n_pids)
            xi_p = pid_masks * (torch.arctanh(beta) ** 2)[:, None]
            # shape: n_pids
            xi_p_norm = torch.sum(xi_p, dim=0)
            # shape: n_pids
            terms = torch.sum(mse[:, None] * xi_p, dim=0)
            return torch.mean(terms / xi_p_norm)
        _ = f"Unknown mode: {self.mode}"
        raise ValueError(_)

    # noinspection PyUnusedLocal
    def forward(
        self,
        *,
        beta: T,
        pred: T,
        particle_id: T,
        track_params: T,
        reconstructable: T,
        **kwargs,
    ) -> T:
        mask = reconstructable > 0
        return self.object_loss(
            pred=pred[mask],
            beta=beta[mask],
            truth=track_params[mask],
            particle_id=particle_id[mask],
        )
