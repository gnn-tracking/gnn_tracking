import torch
from pytorch_lightning.core.mixins import HyperparametersMixin
from torch import Tensor as T
from torch.linalg import norm
from torch_cluster import radius_graph

from gnn_tracking.metrics.losses import MultiLossFct, MultiLossFctReturn
from gnn_tracking.utils.graph_masks import get_good_node_mask_tensors

# ruff: noqa: ARG002


@torch.jit.script
def _hinge_loss_components(
    *,
    x: T,
    att_edges: T,
    rep_edges: T,
    r_emb_hinge: float,
    p_attr: float,
    p_rep: float,
) -> tuple[T, T]:
    eps = 1e-9

    dists_att = norm(x[att_edges[0]] - x[att_edges[1]], dim=-1)
    norm_att = att_edges.shape[1] + eps
    v_att = torch.sum(torch.pow(dists_att, p_attr)) / norm_att

    dists_rep = norm(x[rep_edges[0]] - x[rep_edges[1]], dim=-1)
    norm_rep = rep_edges.shape[1] + eps
    v_rep = r_emb_hinge - torch.sum(torch.pow(dists_rep, p_rep)) / norm_rep

    return v_att, v_rep


class GraphConstructionHingeEmbeddingLoss(MultiLossFct, HyperparametersMixin):
    # noinspection PyUnusedLocal
    def __init__(
        self,
        *,
        lw_repulsive: float = 1.0,
        r_emb: float = 1.0,
        max_num_neighbors: int = 256,
        pt_thld: float = 0.9,
        max_eta: float = 4.0,
        p_attr: float = 1.0,
        p_rep: float = 1.0,
    ):
        """Loss for graph construction using metric learning.

        Args:
            lw_repulsive: Loss weight for repulsive part of potential loss
            r_emb: Radius for edge construction
            max_num_neighbors: Maximum number of neighbors in radius graph building.
                See https://github.com/rusty1s/pytorch_cluster#radius-graph
            pt_thld: pt threshold for particles of interest
            max_eta: maximum eta for particles of interest
            p_attr: Power for the attraction term (default 1: linear loss)
            p_rep: Power for the repulsion term (default 1: linear loss)
        """
        super().__init__()
        self.save_hyperparameters()

    def _get_edges(
        self, *, x: T, batch: T, true_edge_index: T, mask: T, particle_id: T
    ) -> tuple[T, T]:
        """Returns edge index for graph"""
        near_edges = radius_graph(
            x,
            r=self.hparams.r_emb,
            batch=batch,
            loop=False,
            max_num_neighbors=self.hparams.max_num_neighbors,
        )
        # Every edge has to start at a particle of interest, so no special
        # case with noise
        rep_edges = near_edges[:, mask[near_edges[0]]]
        rep_edges = rep_edges[:, particle_id[rep_edges[0]] != particle_id[rep_edges[1]]]
        att_edges = true_edge_index[:, mask[true_edge_index[0]]]
        return att_edges, rep_edges

    # noinspection PyUnusedLocal
    def forward(
        self,
        *,
        x: T,
        particle_id: T,
        batch: T,
        true_edge_index: T,
        pt: T,
        eta: T,
        reconstructable: T,
        **kwargs,
    ) -> MultiLossFctReturn:
        mask = get_good_node_mask_tensors(
            pt=pt,
            particle_id=particle_id,
            reconstructable=reconstructable,
            eta=eta,
            pt_thld=self.hparams.pt_thld,
            max_eta=self.hparams.max_eta,
        )
        att_edges, rep_edges = self._get_edges(
            x=x,
            batch=batch,
            true_edge_index=true_edge_index,
            mask=mask,
            particle_id=particle_id,
        )
        attr, rep = _hinge_loss_components(
            x=x,
            att_edges=att_edges,
            rep_edges=rep_edges,
            r_emb_hinge=self.hparams.r_emb,
            p_attr=self.hparams.p_attr,
            p_rep=self.hparams.p_rep,
        )
        losses = {
            "attractive": attr,
            "repulsive": rep,
        }
        weights: dict[str, float] = {
            "attractive": 1.0,
            "repulsive": self.hparams.lw_repulsive,
        }
        return MultiLossFctReturn(
            loss_dct=losses,
            weight_dct=weights,
        )
