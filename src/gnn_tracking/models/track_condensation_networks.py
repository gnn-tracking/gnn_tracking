"""This module holds the main training models for GNN tracking."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.utils import index_to_mask

from gnn_tracking.models.dynamic_edge_conv import DynamicEdgeConv
from gnn_tracking.models.edge_classifier import ECForGraphTCN, PerfectEdgeClassification
from gnn_tracking.models.interaction_network import InteractionNetwork as IN
from gnn_tracking.models.mlp import MLP
from gnn_tracking.models.resin import ResIN
from gnn_tracking.utils.graph_masks import edge_subgraph


class INConvBlock(nn.Module):
    def __init__(
        self,
        indim,
        h_dim,
        e_dim,
        L,
        k,
        hidden_dim=100,
    ):
        super().__init__()
        self.relu = nn.ReLU()
        self.node_encoder = MLP(2 * indim, h_dim, hidden_dim=hidden_dim, L=1)
        self.edge_conv = DynamicEdgeConv(self.node_encoder, aggr="add", k=k)
        self.edge_encoder = MLP(2 * h_dim, e_dim, hidden_dim=hidden_dim, L=1)
        layers = []
        for _ in range(L):
            layers.append(
                IN(
                    h_dim,
                    e_dim,
                    node_outdim=h_dim,
                    edge_outdim=e_dim,
                    node_hidden_dim=hidden_dim,
                    edge_hidden_dim=hidden_dim,
                )
            )
        self.layers = nn.ModuleList(layers)

    def forward(
        self,
        x: Tensor,
        alpha: float = 0.5,
    ) -> Tensor:
        h, edge_index = self.edge_conv(x)
        h = self.relu(h)
        edge_attr = torch.cat([h[edge_index[0]], h[edge_index[1]]], dim=1)
        edge_attr = self.relu(self.edge_encoder(edge_attr))

        # apply the track condenser
        for layer in self.layers:
            delta_h, edge_attr = layer(h, edge_index, edge_attr)
            h = alpha * h + (1 - alpha) * delta_h
        return h


class PointCloudTCN(nn.Module):
    def __init__(
        self,
        node_indim: int,
        h_dim=10,
        e_dim=10,
        h_outdim=5,
        hidden_dim=100,
        N_blocks=3,
        L=3,
    ):
        """Model to directly process point clouds rather than start with a graph.

        Args:
            node_indim:
            h_dim:   node dimension in latent space
            e_dim: edge dimension in latent space
            h_outdim:  output dimension in clustering space
            hidden_dim:  hidden with of all nn.Linear layers
            N_blocks:  number of edge_conv + IN blocks
            L: message passing depth in each block
        """
        super().__init__()

        layers = [INConvBlock(node_indim, h_dim, e_dim, L=L, k=N_blocks)]
        for i in range(N_blocks):
            layers.append(INConvBlock(h_dim, h_dim, e_dim, L=L, k=N_blocks - i))
        self.layers = nn.ModuleList(layers)

        # modules to predict outputs
        self.B = MLP(h_dim, 1, hidden_dim, L=3)
        self.X = MLP(h_dim, h_outdim, hidden_dim, L=3)

    def forward(
        self,
        data: Data,
        alpha: float = 0.5,
    ) -> dict[str, Tensor]:
        # apply the edge classifier to generate edge weights
        h = data.x
        for layer in self.layers:
            h = layer(h)

        beta = torch.sigmoid(self.B(h))
        # protect against nans
        beta = beta + torch.ones_like(beta) * 10e-12

        h_out = self.X(h)
        return {"W": None, "H": h_out, "B": beta, "P": None}


def mask_nodes_with_few_edges(
    *, n_nodes: int, edge_index: torch.Tensor, min_connections=1
) -> torch.Tensor:
    """Returns a mask where all nodes that do not have at least
    `min_connections` edges are masked.

    Args:
        n_nodes: Total number of hits
        edge_index: Edge index tensor (``2 x n_edges``)
        min_connections: Minimal number of edges that a node should have

    Returns:
        Mask for hits
    """
    if min_connections <= 0:
        return torch.full((n_nodes,), True, dtype=torch.bool, device=edge_index.device)
    node_count_indices, node_counts = torch.unique(
        edge_index.flatten(), return_counts=True
    )
    node_mask = node_counts >= min_connections
    allowed_node_indices = node_count_indices[node_mask]
    node_indices = torch.arange(n_nodes, device=edge_index.device)
    assert len(node_indices) == n_nodes > len(node_count_indices)
    assert n_nodes > node_count_indices.max()
    return torch.isin(node_indices, allowed_node_indices)


class ModularGraphTCN(nn.Module):
    def __init__(
        self,
        *,
        ec: nn.Module,
        hc_in: nn.Module,
        node_indim: int,
        edge_indim: int,
        h_dim=5,
        e_dim=4,
        h_outdim=2,
        hidden_dim=40,
        feed_edge_weights=False,
        ec_threshold=0.5,
        mask_orphan_nodes=False,
        use_ec_embeddings_for_hc=False,
    ):
        """General form of track condensation network based on preconstructed graphs
        with initial step of edge classification (passed as a parameter).

        Args:
            ec: Edge classifier
            hc_in: Track condensor interaction network.
            node_indim: Node feature dimension
            edge_indim: Edge feature dimension
            h_dim: node dimension in the condensation interaction networks
            e_dim: edge dimension in the condensation interaction networks
            h_outdim: output dimension in clustering space
            hidden_dim: width of hidden layers in all perceptrons
            feed_edge_weights: whether to feed edge weights to the track condenser
            ec_threshold: threshold for edge classification
            mask_orphan_nodes: Mask nodes with no connections after EC
            use_ec_embeddings_for_hc: Use edge classifier embeddings as input to
                track condenser. This currently assumes that h_dim and e_dim are
                also the dimensions used in the EC.
        """
        super().__init__()
        self.relu = nn.ReLU()

        #: Edge classification network
        self.ec = ec
        #: Track condensation network (usually made up of interaction networks)
        self.hc_in = hc_in

        if use_ec_embeddings_for_hc:
            node_enc_indim = node_indim + h_dim
            edge_enc_indim = edge_indim + e_dim
        else:
            node_enc_indim = node_indim
            edge_enc_indim = edge_indim + int(feed_edge_weights)

        #: Node encoder network for track condenser
        self.hc_node_encoder = MLP(
            node_enc_indim, h_dim, hidden_dim=hidden_dim, L=2, bias=False
        )
        #: Edge encoder network for track condenser
        self.hc_edge_encoder = MLP(
            edge_enc_indim,
            e_dim,
            hidden_dim=hidden_dim,
            L=2,
            bias=False,
        )

        #: NN to predict beta
        self.p_beta = MLP(h_dim, 1, hidden_dim, L=3)
        #: NN to predict cluster coordinates
        self.p_cluster = MLP(h_dim, h_outdim, hidden_dim, L=3)
        #: NN to predict track parameters
        # self.p_track_param = IN(
        #     node_indim=h_dim,
        #     edge_indim=e_dim + hc_in.length_concatenated_edge_attrs,
        #     node_outdim=1,
        #     edge_outdim=1,
        #     node_hidden_dim=hidden_dim,
        #     edge_hidden_dim=hidden_dim,
        # )
        self._feed_edge_weights = feed_edge_weights
        self.threshold = ec_threshold
        self._mask_orphan_nodes = mask_orphan_nodes
        self._use_ec_embeddings_for_hc = use_ec_embeddings_for_hc

    def forward(
        self,
        data: Data,
    ) -> dict[str, Tensor]:
        ec_result = self.ec(data)
        # Assign all EC  output to the data object, so that the cuts
        # will be applied automatically when we call `data.subgraph(...)` etc.
        data.edge_weights = ec_result["W"]
        data.ec_node_embedding = ec_result.get("node_embedding", None)
        data.ec_edge_embedding = ec_result.get("edge_embedding", None)
        edge_weights_unmasked = data.edge_weights.clone().detach()
        edge_mask = (data.edge_weights > self.threshold).squeeze()
        data = edge_subgraph(data, edge_mask)

        if self._mask_orphan_nodes:
            connected_nodes = data.edge_index.flatten().unique()
            hit_mask = index_to_mask(connected_nodes, size=data.num_nodes)
            data = data.subgraph(connected_nodes)
        else:
            hit_mask = torch.ones(
                data.num_nodes, dtype=torch.bool, device=data.x.device
            )

        # Get the encoded inputs for the track condenser
        edge_attr = data.edge_attr
        x = data.x
        if self._feed_edge_weights:
            edge_attr = torch.concat([data.edge_attr, data.edge_weights], dim=1)
        if self._use_ec_embeddings_for_hc:
            edge_attr = torch.cat([edge_attr, data.ec_edge_embedding], dim=1)
            x = torch.cat([x, data.ec_node_embedding], dim=1)
        h_hc = self.relu(self.hc_node_encoder(x))
        edge_attr_hc = self.relu(self.hc_edge_encoder(edge_attr))

        # Run the track condenser
        h_hc, _, _ = self.hc_in(h_hc, data.edge_index, edge_attr_hc)
        beta = torch.sigmoid(self.p_beta(h_hc))
        # protect against nans
        beta = beta + torch.ones_like(beta) * 10e-9

        h = self.p_cluster(h_hc)
        # track_params, _ = self.p_track_param(
        #     h_hc, data.edge_index, torch.cat(edge_attrs_hc, dim=1)
        # )
        return {
            "W": edge_weights_unmasked,
            "H": h,
            "B": beta,
            "ec_hit_mask": hit_mask,
            "ec_edge_mask": edge_mask,
        }


class GraphTCN(nn.Module):
    def __init__(
        self,
        node_indim: int,
        edge_indim: int,
        *,
        h_dim=5,
        e_dim=4,
        h_outdim=2,
        hidden_dim=40,
        L_ec=3,
        L_hc=3,
        alpha_ec: float = 0.5,
        alpha_hc: float = 0.5,
        **kwargs,
    ):
        """`ModularTCN` with `ECForGraphTCN` as
        edge classification step and several interaction networks as residual layers
        for the track condensor network.

        Args:
            node_indim: Node feature dim
            edge_indim: Edge feature dim
            h_dim: node dimension in latent space
            e_dim: edge dimension in latent space
            h_outdim: output dimension in clustering space
            hidden_dim: width of hidden layers in all perceptrons
            L_ec: message passing depth for edge classifier
            L_hc: message passing depth for track condenser
            alpha_ec: strength of residual connection for multi-layer interaction
                networks in edge classifier
            alpha_hc: strength of residual connection for multi-layer interaction
                networks in track condenser
            **kwargs: Additional keyword arguments passed to `ModularGraphTCN`
        """
        super().__init__()
        ec = ECForGraphTCN(
            node_indim=node_indim,
            edge_indim=edge_indim,
            hidden_dim=hidden_dim,
            interaction_node_dim=h_dim,
            interaction_edge_dim=e_dim,
            L_ec=L_ec,
            alpha=alpha_ec,
        )
        # Todo: Add other resin options
        hc_in = ResIN(
            node_dim=h_dim,
            edge_dim=e_dim,
            object_hidden_dim=hidden_dim,
            relational_hidden_dim=hidden_dim,
            alpha=alpha_hc,
            n_layers=L_hc,
        )
        self._gtcn = ModularGraphTCN(
            ec=ec,
            hc_in=hc_in,
            node_indim=node_indim,
            edge_indim=edge_indim,
            h_dim=h_dim,
            e_dim=e_dim,
            h_outdim=h_outdim,
            hidden_dim=hidden_dim,
            **kwargs,
        )

    def forward(
        self,
        data: Data,
    ) -> dict[str, Tensor]:
        return self._gtcn.forward(data=data)


class PerfectECGraphTCN(nn.Module):
    def __init__(
        self,
        *,
        node_indim: int,
        edge_indim: int,
        h_dim=5,
        e_dim=4,
        h_outdim=2,
        hidden_dim=40,
        L_hc=3,
        alpha_hc: float = 0.5,
        ec_tpr=1.0,
        ec_tnr=1.0,
        **kwargs,
    ):
        """Similar to `GraphTCN` but with a "perfect" (i.e., truth based) edge
        classifier.

        Args:
            node_indim: Node feature dim. Determined by input data.
            edge_indim: Edge feature dim. Determined by input data.
            h_dim: node dimension after encoding
            e_dim: edge dimension after encoding
            h_outdim: output dimension in clustering space
            hidden_dim: dimension of hidden layers in all MLPs used in the interaction
                networks
            L_hc: message passing depth for track condenser
            alpha_hc: strength of residual connection for multi-layer interaction
                networks
            ec_tpr: true positive rate of the perfect edge classifier
            ec_tnr: true negative rate of the perfect edge classifier
            **kwargs: Passed to `ModularGraphTCN`
        """
        super().__init__()
        ec = PerfectEdgeClassification(tpr=ec_tpr, tnr=ec_tnr)
        hc_in = ResIN(
            node_dim=h_dim,
            edge_dim=e_dim,
            object_hidden_dim=hidden_dim,
            relational_hidden_dim=hidden_dim,
            alpha=alpha_hc,
            n_layers=L_hc,
        )
        self._gtcn = ModularGraphTCN(
            ec=ec,
            hc_in=hc_in,
            node_indim=node_indim,
            edge_indim=edge_indim,
            h_dim=h_dim,
            e_dim=e_dim,
            h_outdim=h_outdim,
            hidden_dim=hidden_dim,
            **kwargs,
        )

    def forward(
        self,
        data: Data,
    ) -> dict[str, Tensor]:
        return self._gtcn.forward(data=data)


class PreTrainedECGraphTCN(nn.Module):
    def __init__(
        self,
        ec,
        *,
        node_indim: int,
        edge_indim: int,
        h_dim=5,
        e_dim=4,
        h_outdim=2,
        hidden_dim=40,
        L_hc=3,
        alpha_hc: float = 0.5,
        **kwargs,
    ):
        """GraphTCN for the use with a pre-trained edge classifier

        Args:
            ec: Pre-trained edge classifier
            node_indim: Node feature dim. Determined by input data.
            edge_indim: Edge feature dim. Determined by input data.
            h_dim: node dimension after encoding
            e_dim: edge dimension after encoding
            h_outdim: output dimension in clustering space
            hidden_dim: dimension of hidden layers in all MLPs used in the interaction
                networks
            L_hc: message passing depth for track condenser
            alpha_hc: strength of residual connection for multi-layer interaction
                networks
        """
        super().__init__()
        hc_in = ResIN(
            node_dim=h_dim,
            edge_dim=e_dim,
            object_hidden_dim=hidden_dim,
            relational_hidden_dim=hidden_dim,
            alpha=alpha_hc,
            n_layers=L_hc,
        )
        self._gtcn = ModularGraphTCN(
            ec=ec,
            hc_in=hc_in,
            node_indim=node_indim,
            edge_indim=edge_indim,
            h_dim=h_dim,
            e_dim=e_dim,
            h_outdim=h_outdim,
            hidden_dim=hidden_dim,
            **kwargs,
        )

    def forward(
        self,
        data: Data,
    ) -> dict[str, Tensor]:
        return self._gtcn.forward(data=data)
