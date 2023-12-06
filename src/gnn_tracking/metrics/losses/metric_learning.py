import torch
from pytorch_lightning.core.mixins import HyperparametersMixin
from torch import Tensor as T
from torch import nn
from torch.linalg import norm
from torch.nn.functional import relu
from torch_cluster import radius_graph

# ruff: noqa: ARG002


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
