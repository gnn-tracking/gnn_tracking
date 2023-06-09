from __future__ import annotations

import math

import numpy as np
import torch.nn
from torch import Tensor as T
from torch.nn import Linear, ModuleList, init
from torch.nn.functional import normalize, relu
from torch_cluster import knn_graph
from torch_geometric.data import Data

from gnn_tracking.utils.log import logger


class GraphConstructionFCNN(torch.nn.Module):
    def __init__(
        self,
        *,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        depth: int,
        beta: float = 0.4,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.beta = beta

        self.encoder = Linear(in_dim, hidden_dim, bias=False)
        self.decoder = Linear(hidden_dim, out_dim, bias=False)

        self.layers = ModuleList(
            [Linear(hidden_dim, hidden_dim, bias=False) for _ in range(depth - 1)]
        )
        self.latent_normalization = torch.nn.Parameter(
            torch.Tensor([1.0]), requires_grad=True
        )
        self.reset_parameters()

    def reset_parameters(self):
        self._reset_layer_parameters(self.encoder, var=1 / self.in_dim)
        for layer in self.layers:
            self._reset_layer_parameters(layer, var=2 / self.hidden_dim)
        self._reset_layer_parameters(self.decoder, var=2 / self.hidden_dim)

    @staticmethod
    def _reset_layer_parameters(layer, var: float):
        layer.reset_parameters()
        for p in layer.parameters():
            init.normal_(p.data, mean=0, std=math.sqrt(var))

    def forward(self, data: Data) -> dict[str, T]:
        x = normalize(data.x, p=2.0, dim=1, eps=1e-12, out=None)
        x = self.encoder(x)
        for layer in self.layers:
            x = np.sqrt(self.beta) * layer(relu(x)) + np.sqrt(1 - self.beta) * x
        x = self.decoder(relu(x))
        x *= self.latent_normalization
        return {"H": x}


def knn_with_max_radius(x: T, k: int, max_radius: float | None = None) -> T:
    """A version of kNN that excludes edges with a distance larger than a given radius.

    Args:
        x:
        k: Number of neighbors
        max_radius:

    Returns:
        edge index
    """
    edge_index = knn_graph(x, k=k)
    if max_radius is not None:
        dists = (x[edge_index[0]] - x[edge_index[1]]).norm(dim=-1)
        edge_index = edge_index[:, dists < max_radius]
    return edge_index


class GCWithEF(torch.nn.Module):
    def __init__(
        self,
        ml: torch.nn.Module,
        ef: torch.nn.Module,
        max_radius: float = 1,
        max_num_neighbors: int = 256,
        use_embedding_features=False,
        ratio_of_false=None,
        build_edge_features=False,
    ):
        super().__init__()
        self._ml = ml
        self._ef = ef
        self._max_radius = max_radius
        self._max_num_neighbors = max_num_neighbors
        self._use_embedding_features = use_embedding_features
        self._ratio_of_false = ratio_of_false
        self._build_edge_features = build_edge_features
        if build_edge_features and ratio_of_false:
            logger.warning(
                "Subsampling false edges. This might not make sense"
                " for message passing."
            )

    def forward(self, data: Data):
        mo = self._ml(data)
        edge_index = knn_with_max_radius(
            mo["H"], max_radius=self._max_radius, k=self._max_num_neighbors
        )
        y: T = (  # type: ignore
            data.particle_id[edge_index[0]] == data.particle_id[edge_index[1]]
        )
        if not self._use_embedding_features:
            x = data.x
        else:
            x = torch.cat((mo["H"], data.x), dim=1)
        # print(edge_index.shape, )
        if self._ratio_of_false and self.training:
            num_true = y.sum()
            num_false_to_keep = int(num_true * self._ratio_of_false)
            false_edges = edge_index[:, ~y][:, :num_false_to_keep]
            true_edges = edge_index[:, y]
            edge_index = torch.cat((false_edges, true_edges), dim=1)
            y = torch.cat(
                (
                    torch.zeros(false_edges.shape[1], device=y.device),
                    torch.ones(true_edges.shape[1], device=y.device),
                )
            )
        # print(false_edges.shape, true_edges.shape, edge_index.shape, y.shape)
        edge_features = None
        if self._build_edge_features:
            edge_features = torch.cat((x[edge_index[0]], x[edge_index[1]]), dim=1)
        graph = Data(
            x=x,
            edge_index=edge_index,
            y=y,
            pt=data.pt,
            particle_id=data.particle_id,
            sector=data.sector,
            reconstructable=data.reconstructable,
            edge_attr=edge_features,
        )
        assert len(graph.y) == graph.edge_index.shape[1]
        return self._ef(graph) | {
            "y": y.float(),
            "edge_index": edge_index,
        }
