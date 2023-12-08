from typing import Any

import numpy as np
import torch
from pytorch_lightning.core.mixins import HyperparametersMixin
from torch import Tensor as T
from torch.nn.functional import mse_loss
from torch_cluster import radius_graph

from gnn_tracking.metrics.losses import MultiLossFct, MultiLossFctReturn
from gnn_tracking.utils.graph_masks import get_good_node_mask_tensors
from gnn_tracking.utils.log import logger

# ruff: noqa: ARG002


def _first_occurrences(input_array: T) -> T:
    """Return the first occurrence of each unique element in a 1D array"""
    return torch.tensor(
        np.unique(input_array.cpu(), return_index=True)[1], device=input_array.device
    )


def _square_distances(edges: T, positions: T) -> T:
    """Returns squared distances between two sets of points"""
    return torch.sum((positions[edges[0]] - positions[edges[1]]) ** 2, dim=-1)


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
    """Extracted function for condensation loss. See `PotentialLoss` for details."""
    # For better readability, variables that are only relevant in one "block"
    # are prefixed with an underscore
    # _j means indexed as hits (... x n_hits)
    # _k means indexed as objects (... x n_objects_of_interest)
    # _e means indexed by edge
    # where n_objects_of_interest = len(unique(particle_id[mask]))

    # -- 1. Determine indices of condensation points (CPs) and q --
    _sorted_indices_j = torch.argsort(beta, descending=True)
    _pids_sorted = particle_id[_sorted_indices_j]
    _alphas = _sorted_indices_j[_first_occurrences(_pids_sorted)]
    # Index of condensation points in node array
    alphas_k = _alphas[particle_id[_alphas] > 0]
    assert alphas_k.size()[0] > 0, "No particles found, cannot evaluate loss"
    # "Charge"
    q_j = torch.arctanh(beta) ** 2 + q_min
    assert not torch.isnan(q_j).any(), "q contains NaNs"

    # -- 2. Edges for repulsion loss --
    _radius_edges = radius_graph(
        x=x, r=radius_threshold, max_num_neighbors=max_num_neighbors, loop=False
    )
    # Now filter out everything that doesn't include a CP or connects two hits of the
    # same particle
    _to_cp_e = torch.isin(_radius_edges[0], alphas_k)
    _is_repulsive_e = particle_id[_radius_edges[0]] != particle_id[_radius_edges[1]]
    repulsion_edges_e = _radius_edges[:, _is_repulsive_e & _to_cp_e]

    # -- 3. Edges for attractive loss --
    # 1D array (n_nodes): 1 for CPs, 0 otherwise
    is_cp_j = torch.zeros(len(particle_id), dtype=bool, device=x.device).scatter_(
        0, alphas_k, 1
    )
    # hit-indices of all non-CPs
    _non_cp_indices = torch.arange(len(particle_id), device=x.device)[~is_cp_j]
    # for each non-CP hit, the index of the corresponding CP
    corresponding_alpha = _alphas[
        torch.searchsorted(particle_id[_alphas], particle_id[_non_cp_indices])
    ]
    # Insert alpha indices into their respective positions to form attraction edges
    _attraction_edges_e = (
        torch.arange(len(particle_id), device=x.device).unsqueeze(0).repeat(2, 1)
    )
    _attraction_edges_e[1, ~is_cp_j] = corresponding_alpha
    # Apply mask to attraction edges
    attraction_edges_e = _attraction_edges_e[:, mask]

    # -- 4. Calculate loss --
    # Protect against sqrt not being differentiable around 0
    eps = 1e-9
    repulsion_distances_e = radius_threshold - torch.sqrt(
        eps + _square_distances(repulsion_edges_e, x)
    )
    attraction_distances_e = _square_distances(attraction_edges_e, x)

    va = (
        attraction_distances_e * q_j[attraction_edges_e[0]] * q_j[attraction_edges_e[1]]
    )
    vr = repulsion_distances_e * q_j[repulsion_edges_e[0]] * q_j[repulsion_edges_e[1]]

    if torch.isnan(vr).any():
        vr = torch.tensor([[0.0]])
        logger.warning("Repulsive loss is NaN")

    hit_is_noise_j = particle_id == 0

    losses = {
        "attractive": (1 / mask.sum()) * torch.sum(va),
        "repulsive": (1 / x.size()[0]) * torch.sum(vr),
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
    # To protect against nan in divisions
    eps = 1e-9

    # x: n_nodes x n_outdim
    not_noise = object_id > noise_threshold
    unique_oids = torch.unique(object_id[object_mask])
    assert len(unique_oids) > 0, "No particles found, cannot evaluate loss"
    # n_nodes x n_pids
    # The nodes in every column correspond to the hits of a single particle and
    # should attract each other
    attractive_mask = object_id.view(-1, 1) == unique_oids.view(1, -1)

    q = torch.arctanh(beta) ** 2 + q_min
    assert not torch.isnan(q).any(), "q contains NaNs"
    # n_objs
    alphas = torch.argmax(q.view(-1, 1) * attractive_mask, dim=0)

    # _j means indexed by hits
    # _k means indexed by objects

    # n_objs x n_outdim
    x_k = x[alphas]
    # 1 x n_objs
    q_k = q[alphas].view(1, -1)

    dist_j_k = torch.cdist(x, x_k)

    qw_j_k = q.view(-1, 1) * q_k

    att_norm_k = (attractive_mask.sum(dim=0) + eps) * len(unique_oids)
    qw_att = (qw_j_k / att_norm_k)[attractive_mask]

    # Attractive potential/loss
    v_att = (qw_att * torch.square(dist_j_k[attractive_mask])).sum()

    repulsive_mask = (~attractive_mask) & (dist_j_k < 1)
    n_rep_k = (~attractive_mask).sum(dim=0)
    n_rep = repulsive_mask.sum()
    # Don't normalize to repulsive_mask, it includes the dist < 1 count,
    # (less points within the radius 1 ball should translate to lower loss)
    rep_norm = (n_rep_k + eps) * len(unique_oids)
    if n_rep > max_n_rep > 0:
        sampling_freq = max_n_rep / n_rep
        sampling_mask = (
            torch.rand_like(repulsive_mask, dtype=torch.float16) < sampling_freq
        )
        repulsive_mask &= sampling_mask
        rep_norm *= sampling_freq
    qw_rep = (qw_j_k / rep_norm)[repulsive_mask]
    v_rep = (qw_rep * (1 - dist_j_k[repulsive_mask])).sum()

    l_coward = torch.mean(1 - beta[alphas])
    l_noise = torch.mean(beta[~not_noise])

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


# _first_occurrences prevents jit
# @torch.jit.script
def _background_loss(*, beta: T, particle_id: T) -> dict[str, T]:
    """Extracted function for JIT-compilation."""
    sorted_indices = torch.argsort(beta, descending=True)
    ids_sorted = particle_id[sorted_indices]
    noisy_alphas = sorted_indices[_first_occurrences(ids_sorted)]
    alphas = noisy_alphas[particle_id[noisy_alphas] > 0]

    beta_alphas = beta[alphas]
    coward_loss = torch.mean(1 - beta_alphas)
    noise_mask = particle_id == 0
    noise_loss = torch.Tensor([0.0])
    if noise_mask.any():
        noise_loss = torch.mean(beta[noise_mask])
    return {
        "coward": coward_loss,
        "noise": noise_loss,
    }


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
