"""This module contains loss functions for the GNN tracking model."""

# Ignore unused arguments because of save_hyperparameters
# ruff: noqa: ARG002

import copy
import math
from abc import ABC, abstractmethod
from typing import Any, Mapping, Protocol, Union

import numpy as np
import torch
from pytorch_lightning.core.mixins import HyperparametersMixin
from torch import Tensor, nn
from torch import Tensor as T
from torch.linalg import norm
from torch.nn.functional import binary_cross_entropy, mse_loss, relu
from torch_cluster import radius_graph

from gnn_tracking.utils.graph_masks import get_good_node_mask_tensors
from gnn_tracking.utils.log import logger


@torch.jit.script
def _binary_focal_loss(
    *,
    inpt: T,
    target: T,
    alpha: float,
    gamma: float,
    pos_weight: T,
) -> T:
    """Extracted function for JIT compilation."""
    probs_pos = inpt
    probs_neg = 1 - inpt

    pos_term = -alpha * pos_weight * probs_neg.pow(gamma) * target * probs_pos.log()
    neg_term = -(1.0 - alpha) * probs_pos.pow(gamma) * (1.0 - target) * probs_neg.log()
    loss_tmp = pos_term + neg_term

    return torch.mean(loss_tmp)


# Follows the implementation in kornia at
# https://github.com/kornia/kornia/blob/master/kornia/losses/focal.py
# (binary_focal_loss_with_logits function)
def binary_focal_loss(
    *,
    inpt: T,
    target: T,
    alpha: float = 0.25,
    gamma: float = 2.0,
    pos_weight: T | None = None,
) -> T:
    """Binary Focal Loss, following https://arxiv.org/abs/1708.02002.

    Args:
        inpt:
        target:
        alpha: Weight for positive/negative results
        gamma: Focusing parameter
        pos_weight: Can be used to balance precision/recall
    """
    if pos_weight is None:
        pos_weight = torch.tensor([1.0], device=inpt.device)

    assert gamma >= 0.0
    assert 0 <= alpha <= 1
    assert not torch.isnan(inpt).any()
    assert not torch.isnan(target).any()
    assert pos_weight is not None

    # JIT compilation does not support optional arguments
    return _binary_focal_loss(
        inpt=inpt,
        target=target,
        alpha=alpha,
        gamma=gamma,
        pos_weight=pos_weight,
    )


def falsify_low_pt_edges(
    *, y: T, edge_index: T | None = None, pt: T | None = None, pt_thld: float = 0.0
) -> T:
    """Modify the ground truth to-be-predicted by the edge classification
    to consider edges that include a hit with pt < pt_thld as false.

    Args:
        y: True classification
        edge_index:
        pt: Hit pt
        pt_thld: Apply pt threshold

    Returns:
        True classification with additional criteria applied
    """
    if math.isclose(pt_thld, 0.0):
        return y
    assert edge_index is not None
    assert pt is not None
    # Because false edges are already falsified, we can
    # it's enough to check the first hit of the edge for its pt
    return y.bool() & (pt[edge_index[0, :]] > pt_thld)


class FalsifyLowPtEdgeWeightLoss(torch.nn.Module, ABC, HyperparametersMixin):
    # noinspection PyUnusedLocal
    def __init__(self, *, pt_thld: float = 0.0):
        """Add an option to falsify edges with low pt to edge classification losses."""
        super().__init__()
        self.save_hyperparameters()

    # noinspection PyUnusedLocal
    def forward(
        self, *, w: T, y: T, edge_index: T | None = None, pt: T | None = None, **kwargs
    ) -> T:
        y = falsify_low_pt_edges(
            y=y, edge_index=edge_index, pt=pt, pt_thld=self.hparams.pt_thld
        )
        return self._forward(y=y.float(), w=w)

    @abstractmethod
    def _forward(self, *, w: T, y: T, **kwargs) -> T:
        pass


class EdgeWeightBCELoss(FalsifyLowPtEdgeWeightLoss):
    """Binary Cross Entropy loss function for edge classification"""

    @staticmethod
    def _forward(*, w: T, y: T, **kwargs) -> T:
        return binary_cross_entropy(w, y, reduction="mean")


class EdgeWeightFocalLoss(FalsifyLowPtEdgeWeightLoss):
    # noinspection PyUnusedLocal
    def __init__(
        self,
        *,
        alpha=0.25,
        gamma=2.0,
        pos_weight=None,
        **kwargs,
    ):
        """Loss function based on focal loss for edge classification.
        See `binary_focal_loss` for details.
        """
        super().__init__(**kwargs)
        self.save_hyperparameters()

    def _forward(self, *, w: T, y: T, **kwargs) -> T:
        pos_weight = self.hparams.pos_weight
        if pos_weight is None:
            pos_weight = torch.tensor([1.0], device=w.device)
        return binary_focal_loss(
            inpt=w,
            target=y,
            alpha=self.hparams.alpha,
            gamma=self.hparams.gamma,
            pos_weight=pos_weight,
        )


class HaughtyFocalLoss(torch.nn.Module, HyperparametersMixin):
    def __init__(
        self,
        *,
        alpha: float = 0.25,
        gamma: float = 2.0,
        pt_thld=0.0,
    ):
        super().__init__()
        self.save_hyperparameters()
        self._alpha = alpha
        self._gamma = gamma
        self._pt_thld = pt_thld

    # noinspection PyUnusedLocal
    def forward(self, *, w: T, y: T, edge_index: T, pt: T, **kwargs) -> T:
        pos_weight = falsify_low_pt_edges(
            y=y, edge_index=edge_index, pt=pt, pt_thld=self._pt_thld
        )
        return binary_focal_loss(
            inpt=w,
            target=y.long(),
            alpha=self._alpha,
            gamma=self._gamma,
            pos_weight=pos_weight,
        )


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
) -> dict[str, T]:
    """Extracted function for condensation loss. See `PotentialLoss` for details."""
    # For better readability, variables that are only relevant in one "block"
    # are prefixed with an underscore

    # -- 1. Determine indices of condensation points (CPs) and q --
    _sorted_indices = torch.argsort(beta, descending=True)
    _pids_sorted = particle_id[_sorted_indices]
    _alphas = _sorted_indices[_first_occurrences(_pids_sorted)]
    # Index of condensation points in node array
    alphas = _alphas[particle_id[_alphas] > 0]
    assert alphas.size()[0] > 0, "No particles found, cannot evaluate loss"
    q = torch.arctanh(beta) ** 2 + q_min
    assert not torch.isnan(q).any(), "q contains NaNs"

    # -- 2. Edges for repulsion loss --
    _radius_edges = radius_graph(
        x=x, r=radius_threshold, max_num_neighbors=max_num_neighbors, loop=False
    )
    # Now filter out everything that doesn't include a CP or connects two hits of the
    # same particle
    _to_cp = torch.isin(_radius_edges[0], alphas)
    _is_repulsive = particle_id[_radius_edges[0]] != particle_id[_radius_edges[1]]
    repulsion_edges = _radius_edges[:, _is_repulsive & _to_cp]

    # -- 3. Edges for attractive loss --
    # 1D array (n_nodes): 1 for CPs, 0 otherwise
    alpha_hits_filter = torch.zeros(
        len(particle_id), dtype=bool, device=x.device
    ).scatter_(0, alphas, 1)
    # indices of all non-CPs
    non_alpha_indices = torch.arange(len(particle_id), device=x.device)[
        ~alpha_hits_filter
    ]

    # for each non-CP hit, the index of the corresponding CP
    alpha_indices = _alphas[
        torch.searchsorted(particle_id[_alphas], particle_id[non_alpha_indices])
    ]

    # Insert alpha indices into their respective positions to form attraction edges
    unmasked_attraction_edges = (
        torch.arange(len(particle_id), device=x.device).unsqueeze(0).repeat(2, 1)
    )
    # fixme: What about the case where there are no associated CPs (noise hits etc)?
    unmasked_attraction_edges[1, ~alpha_hits_filter] = alpha_indices

    # Apply mask to attraction edges
    attraction_edges = unmasked_attraction_edges[:, mask]

    # -- 4. Calculate loss --
    repulsion_distances = radius_threshold - torch.sqrt(
        _square_distances(repulsion_edges, x)
    )
    attraction_distances = _square_distances(attraction_edges, x)

    va = attraction_distances * q[attraction_edges[0]] * q[attraction_edges[1]]
    vr = repulsion_distances * q[repulsion_edges[0]] * q[repulsion_edges[1]]

    if torch.isnan(vr).any():
        vr = torch.tensor([[0.0]])
        logger.warning("Repulsive loss is NaN")

    return {
        "attractive": (1 / mask.sum()) * torch.sum(va),
        "repulsive": (1 / x.size()[0]) * torch.sum(vr),
        "noise": torch.tensor([0.0], device=beta.device),  # todo
        "coward": torch.tensor([0.0], device=beta.device),  # todo
    }


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
) -> dict[str, T]:
    """Extracted function for torch compilation. See `condensation_loss_tiger` for
    docstring.

    Args:
        object_mask: Mask for the particles that should be considered for the loss
            this is broadcased to n_hits
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
    # todo: Should we use object_mask instead of not noise?
    l_noise = torch.mean(beta[~not_noise])

    return {
        "attractive": v_att,
        "repulsive": v_rep,
        "coward": l_coward,
        "noise": l_noise,
        "n_rep": n_rep,
    }


class CondensationLossRG(torch.nn.Module, HyperparametersMixin):
    def __init__(
        self,
        *,
        q_min: float = 0.01,
        pt_thld: float = 0.9,
        max_eta: float = 4.0,
        max_num_neighbors: int = 256,
        sample_pids: float = 1.0,
    ):
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
    ) -> dict[str, T]:
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
        return _radius_graph_condensation_loss(
            beta=beta,
            x=x,
            particle_id=particle_id,
            mask=mask,
            q_min=self.hparams.q_min,
            radius_threshold=1.0,
            max_num_neighbors=self.hparams.max_num_neighbors,
        )


class CondensationLossTiger(torch.nn.Module, HyperparametersMixin):
    def __init__(
        self,
        *,
        q_min: float = 0.01,
        pt_thld: float = 0.9,
        max_eta: float = 4.0,
        max_n_rep: int = 0,
        sample_pids: float = 1.0,
    ):
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
    ) -> dict[str, T]:
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
        return condensation_loss_tiger(
            beta=beta,
            x=x,
            object_id=particle_id,
            object_mask=mask,
            q_min=self.hparams.q_min,
            noise_threshold=0.0,
            max_n_rep=self.hparams.max_n_rep,
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


class LossFctType(Protocol):
    """Type of a loss function"""

    def __call__(self, *args: Any, **kwargs: Any) -> Tensor:
        ...

    def to(self, device: torch.device) -> "LossFctType":
        ...


loss_weight_type = Union[float, dict[str, float], list[float]]


def unpack_loss_returns(key: str, returns: Any) -> dict[str, float | Tensor]:
    """Some of our loss functions return a dictionary or a list of individual losses.
    This function unpacks these into a dictionary of individual losses with appropriate
    keys.

    Args:
        key: str (name of the loss function)
        returns: dict or list or single value

    Returns:
        dict of individual losses
    """
    if isinstance(returns, Mapping):
        return {f"{key}_{k}": v for k, v in returns.items()}
    if isinstance(returns, (list, tuple)):
        # Don't put 'Sequence' here, because Tensors are Sequences
        return {f"{key}_{i}": v for i, v in enumerate(returns)}
    return {key: returns}


class LossClones(torch.nn.Module):
    def __init__(self, loss: torch.nn.Module, prefixes=("w", "y")) -> None:
        """Wrapper for a loss function that evaluates it on multiple inputs.
        The forward method will look for all model outputs that start with `w_`
        (or another specified prefix) and evaluate the loss function for each of them,
        returning a dictionary of losses (with keys equal to the suffixes).

        Usage example 1:

        .. code-block:: python

            losses = {
                "potential": (PotentialLoss(), 1.),
                "edge": (LossClones(EdgeWeightBCELoss()), [1.0, 2.0, 3.0])
            }

        will evaluate three clones of the BCE loss function, one for each EC layer.

        Usage Example 2:


        .. code-block:: python

            losses = {
                "potential": (PotentialLoss(), 1.),
                "edge": (LossClones(EdgeWeightBCELoss()), {}))
            }

        this works with a variable number of layers. The weights are all 1.

        Under the hood, ``ECLossClones(EdgeWeightBCELoss())(model output)`` will output
        a dictionary of the individual losses, keyed by their suffixes (in a similar
        way to how `PotentialLoss` returns a dictionary of losses).

        Args:
            loss: Loss function to be evaluated on multiple inputs
            prefixes: Prefixes of the model outputs that should be evaluated.
                An underscore is assumed (set prefix to `w` for `w_0`, `w_1`, etc.)

        """
        super().__init__()
        self._loss = loss
        self._prefixes = prefixes

    def forward(self, **kwargs) -> dict[str, Tensor]:
        kwargs = copy.copy(kwargs)
        for prefix in self._prefixes:
            if prefix in kwargs:
                logger.warning(
                    f"LossClones prefix {prefix} is also a model output. Removing "
                    f"this for now, but you probably want to clean up if this is not "
                    f"intended."
                )
                kwargs.pop(prefix)
        losses = {}
        ec_layer_names = sorted(
            [
                k[len(self._prefixes[0]) + 1 :]
                for k in kwargs
                if k.startswith(self._prefixes[0] + "_")
            ]
        )
        for layer_name in ec_layer_names:
            rename_dct = {f"{prefix}_{layer_name}": prefix for prefix in self._prefixes}
            renamed_kwargs = {rename_dct.get(k, k): v for k, v in kwargs.items()}
            loss = self._loss(**renamed_kwargs)
            losses[layer_name] = loss
        return losses


@torch.jit.script
def _hinge_loss_components(
    *,
    x: T,
    edge_index: T,
    particle_id: T,
    pt: T,
    r_emb_hinge: float,
    pt_thld: float,
    p_attr: float,
    p_rep: float,
) -> tuple[T, T]:
    true_edge = (particle_id[edge_index[0]] == particle_id[edge_index[1]]) & (
        particle_id[edge_index[0]] > 0
    )
    true_high_pt_edge = true_edge & (pt[edge_index[0]] > pt_thld)
    dists = norm(x[edge_index[0]] - x[edge_index[1]], dim=-1)
    normalization = true_high_pt_edge.sum() + 1e-8
    return torch.sum(
        torch.pow(dists[true_high_pt_edge], p_attr)
    ) / normalization, torch.sum(
        relu(r_emb_hinge - torch.pow(dists[~true_edge], p_rep)) / normalization
    )


class GraphConstructionHingeEmbeddingLoss(nn.Module, HyperparametersMixin):
    # noinspection PyUnusedLocal
    def __init__(
        self,
        *,
        r_emb=1,
        max_num_neighbors: int = 256,
        attr_pt_thld: float = 0.9,
        p_attr: float = 1,
        p_rep: float = 1,
    ):
        """Loss for graph construction using metric learning.

        Args:
            r_emb: Radius for edge construction
            max_num_neighbors: Maximum number of neighbors in radius graph building.
                See https://github.com/rusty1s/pytorch_cluster#radius-graph
            p_attr: Power for the attraction term (default 1: linear loss)
            p_rep: Power for the repulsion term (default 1: linear loss)
        """
        super().__init__()
        self.save_hyperparameters()

    def _build_graph(self, x: T, batch: T, true_edge_index: T, pt: T) -> T:
        true_edge_mask = pt[true_edge_index[0]] > self.hparams.attr_pt_thld
        near_edges = radius_graph(
            x,
            r=self.hparams.r_emb,
            batch=batch,
            loop=False,
            max_num_neighbors=self.hparams.max_num_neighbors,
        )
        return torch.unique(
            torch.cat([true_edge_index[:, true_edge_mask], near_edges], dim=-1), dim=-1
        )

    # noinspection PyUnusedLocal
    def forward(
        self, *, x: T, particle_id: T, batch: T, true_edge_index: T, pt: T, **kwargs
    ) -> dict[str, T]:
        edge_index = self._build_graph(
            x=x, batch=batch, true_edge_index=true_edge_index, pt=pt
        )
        attr, rep = _hinge_loss_components(
            x=x,
            edge_index=edge_index,
            particle_id=particle_id,
            r_emb_hinge=self.hparams.r_emb,
            pt=pt,
            pt_thld=self.hparams.attr_pt_thld,
            p_attr=self.hparams.p_attr,
            p_rep=self.hparams.p_rep,
        )
        return {
            "attractive": attr,
            "repulsive": rep,
        }
