"""Models for embeddings used for graph construction."""


# Ignore unused arguments because of save_hyperparameters
# ruff: noqa: ARG002

import math

import numpy as np
import torch.nn
from pytorch_lightning.core.mixins import HyperparametersMixin
from torch import Tensor as T
from torch import nn
from torch.nn import Linear, ModuleList, init
from torch.nn.functional import normalize, relu
from torch_cluster import knn_graph
from torch_geometric.data import Data

from gnn_tracking.utils.asserts import assert_feat_dim
from gnn_tracking.utils.lightning import get_model, obj_from_or_to_hparams
from gnn_tracking.utils.log import logger
from gnn_tracking.utils.torch_utils import freeze_if


class GraphConstructionFCNN(nn.Module, HyperparametersMixin):
    # noinspection PyUnusedLocal
    def __init__(
        self,
        *,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        depth: int,
        beta: float = 0.4,
    ):
        """Metric learning embedding fully connected NN.

        Args:
            in_dim: Input dimension
            hidden_dim: Hidden dimension
            out_dim: Output dimension = embedding space
            depth: Number of layers
            beta: Strength of residual connections
        """

        super().__init__()
        self.save_hyperparameters()

        self._encoder = Linear(in_dim, hidden_dim, bias=False)
        self._decoder = Linear(hidden_dim, out_dim, bias=False)

        self._layers = ModuleList(
            [Linear(hidden_dim, hidden_dim, bias=False) for _ in range(depth - 1)]
        )
        self._latent_normalization = torch.nn.Parameter(
            torch.Tensor([1.0]), requires_grad=True
        )
        self.reset_parameters()

    def reset_parameters(self):
        self._reset_layer_parameters(self._encoder, var=1 / self.hparams.in_dim)
        for layer in self._layers:
            self._reset_layer_parameters(layer, var=2 / self.hparams.hidden_dim)
        self._reset_layer_parameters(self._decoder, var=2 / self.hparams.hidden_dim)

    @staticmethod
    def _reset_layer_parameters(layer, var: float):
        layer.reset_parameters()
        for p in layer.parameters():
            init.normal_(p.data, mean=0, std=math.sqrt(var))

    def forward(self, data: Data) -> dict[str, T]:
        assert_feat_dim(data.x, self.hparams.in_dim)
        x = normalize(data.x, p=2.0, dim=1, eps=1e-12, out=None)
        x = self._encoder(x)
        for layer in self._layers:
            x = (
                np.sqrt(self.hparams.beta) * layer(relu(x))
                + np.sqrt(1 - self.hparams.beta) * x
            )
        x = self._decoder(relu(x))
        x *= self._latent_normalization
        assert x.shape[1] == self.hparams.out_dim
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


class MLGraphConstruction(nn.Module, HyperparametersMixin):
    # noinspection PyUnusedLocal
    def __init__(
        self,
        ml: torch.nn.Module | None = None,
        *,
        ec: torch.nn.Module | None = None,
        max_radius: float = 1,
        max_num_neighbors: int = 256,
        use_embedding_features=False,
        ratio_of_false=None,
        build_edge_features=True,
        ec_threshold=None,
        ml_freeze: bool = True,
        ec_freeze: bool = True,
        embedding_slice: tuple[int, int] | None = None,
    ):
        """Builds graph from embedding space. If you want to start from a checkpoint,
        use `MLGraphConstruction.from_chkpt`.

        Args:
            ml: Metric learning embedding module. If not specified, it is assumed that
                the node features from the data object are already the embedding
                coordinates. To use a subset of the embedding coordinates, use
                ``embedding_slice``.
            ec: Directly apply edge filter
            max_radius: Maximum radius for kNN
            max_num_neighbors: Number of neighbors for kNN
            use_embedding_features: Add embedding space features to node features
            ratio_of_false: Subsample false edges
            build_edge_features:
            ec_threshold:
            embedding_slice: Used if ``ml`` is None. If not None, all node features
                are used. If a tuple, the first element is the start index and the
                second element is the end index.
        """
        super().__init__()
        self.save_hyperparameters(ignore=["ml", "ec"])
        self._ml = freeze_if(obj_from_or_to_hparams(self, "ml", ml), ml_freeze)
        self._ef = freeze_if(obj_from_or_to_hparams(self, "ec", ec), ec_freeze)
        if self._ef is not None and ec_threshold is None:
            msg = "ec_threshold must be set if ec/ef is not None"
            raise ValueError(msg)
        if build_edge_features and ratio_of_false:
            logger.warning(
                "Subsampling false edges. This might not make sense"
                " for message passing."
            )

    @classmethod
    def from_chkpt(
        cls,
        ml_chkpt_path: str = "",
        ec_chkpt_path: str = "",
        *,
        ml_class_name: str = "gnn_tracking.training.ml.MLModule",
        ec_class_name: str = "gnn_tracking.training.ec.ECModule",
        **kwargs,
    ) -> "MLGraphConstruction":
        """Build `MLGraphConstruction` from checkpointed models.

        Args:
            ml_chkpt_path: Path to metric learning checkpoint
            ec_chkpt_path: Path to edge filter checkpoint. If empty, no EC will be
                used.
            ml_class_name: Class name of metric learning lightning module
                (default should almost always be fine)
            ec_class_name: Class name of edge filter lightning module
                (default should almost always be fine)
            **kwargs: Additional arguments passed to `MLGraphConstruction`
        """
        ml = get_model(ml_class_name, ml_chkpt_path)
        ec = get_model(ec_class_name, ec_chkpt_path)
        return cls(
            ml=ml,
            ec=ec,
            **kwargs,
        )

    @property
    def out_dim(self) -> tuple[int, int]:
        """Returns node, edge, output dims"""
        node_dim = self._ml.in_dim
        if self.hparams.use_embedding_features:
            node_dim += self._ml.out_dim
        edge_dim = 2 * node_dim if self.hparams.build_edge_features else 0
        return node_dim, edge_dim

    def forward(self, data: Data) -> Data:
        if self._ml is not None:
            mo = self._ml(data)
            embedding_features = mo["H"]
        else:
            embedding_features = data.x[slice(self.hparams.embedding_slice)]
        edge_index = knn_with_max_radius(
            embedding_features,
            max_radius=self.hparams.max_radius,
            k=self.hparams.max_num_neighbors,
        )
        y: T = (  # type: ignore
            data.particle_id[edge_index[0]] == data.particle_id[edge_index[1]]
        )
        if not self.hparams.use_embedding_features:
            x = data.x
        else:
            x = torch.cat((mo["H"], data.x), dim=1)
        # print(edge_index.shape, )
        if self.hparams.ratio_of_false and self.training:
            num_true = y.sum()
            num_false_to_keep = int(num_true * self.hparams.ratio_of_false)
            false_edges = edge_index[:, ~y][:, :num_false_to_keep]
            true_edges = edge_index[:, y]
            edge_index = torch.cat((false_edges, true_edges), dim=1)
            y = torch.cat(
                (
                    torch.zeros(false_edges.shape[1], device=y.device),
                    torch.ones(true_edges.shape[1], device=y.device),
                )
            )
        edge_features = None
        if self.hparams.build_edge_features:
            edge_features = torch.cat(
                (
                    x[edge_index[0]] - x[edge_index[1]],
                    x[edge_index[0]] + x[edge_index[1]],
                ),
                dim=1,
            )
        if self._ef is not None:
            w = self._ef(edge_features)["W"]
            edge_index = edge_index[:, w > self.hparams.ef_threshold]
        return Data(
            x=x,
            edge_index=edge_index,
            y=y.long(),
            pt=data.pt,
            particle_id=data.particle_id,
            sector=data.sector,
            reconstructable=data.reconstructable,
            edge_attr=edge_features,
        )


class MLGraphConstructionFromChkpt(nn.Module, HyperparametersMixin):
    def __new__(cls, *args, **kwargs) -> MLGraphConstruction:
        """Alias for `MLGraphConstruction.from_chkpt` for use in yaml files"""
        return MLGraphConstruction.from_chkpt(*args, **kwargs)


class MLPCTransformer(nn.Module, HyperparametersMixin):
    # noinspection PyUnusedLocal
    def __init__(
        self,
        model: nn.Module,
        *,
        original_features: bool = False,
        freeze: bool = True,
    ):
        """Transforms a point cloud (PC) using a metric learning (ML) model.
        This is just a thin wrapper around the ML module with specification of what
        to do with the resulting latent space.
        In contrast to `MLGraphConstructionFromChkpt`, this class does not build a
        graph from the latent space but returns a transformed point cloud.
        Use `MLPCTransformer.from_ml_chkpt` to build from a checkpointed ML model.

        .. warning::
            In the current implementation, the original ``Data`` object is modified
            by `forward`.

        Args:
            model: Metric learning model. Should return latent space with key "H"
            original_features: Include original node features as node features (after
                the transformed ones)
        """
        super().__init__()
        self._ml = freeze_if(obj_from_or_to_hparams(self, "ml", model), freeze)
        self.save_hyperparameters(ignore=["model"])

    @classmethod
    def from_ml_chkpt(
        cls,
        chkpt_path: str,
        *,
        class_name: str = "gnn_tracking.training.ml.MLModule",
        **kwargs,
    ):
        """Build `MLPCTransformer` from checkpointed ML model.

        Args:
            chkpt_path: Path to checkpoint
            class_name: Lightning module class name that was used for training.
                Probably default covers most cases.
            **kwargs: Additional kwargs passed to `MLPCTransformer` constructor
        """
        ml_model = get_model(class_name, chkpt_path)
        return cls(
            ml_model,
            **kwargs,
        )

    def forward(self, data: Data) -> Data:
        out = self._ml(data)
        if self.hparams.original_features:
            data.x = torch.cat((out["H"], data.x), dim=1)
        else:
            data.x = out["H"]
        return data


class MLPCTransformerFromMLChkpt(MLPCTransformer):
    def __new__(*args, **kwargs) -> MLPCTransformer:
        """Alias for `MLPCTransformer.from_ml_chkpt` for use in yaml configs"""
        return MLPCTransformer.from_chkpt(*args, **kwargs)
