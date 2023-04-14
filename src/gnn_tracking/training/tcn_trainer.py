from __future__ import annotations

import collections
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path, PurePath
from typing import Any, Callable, DefaultDict, Mapping

import numpy as np
import pandas as pd
import tabulate
import torch
from torch import Tensor
from torch.optim import Adam
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.utils import mask_to_index

from gnn_tracking.metrics.binary_classification import (
    BinaryClassificationStats,
    get_maximized_bcs,
    roc_auc_score,
)
from gnn_tracking.metrics.losses import LossFctType
from gnn_tracking.postprocessing.clusterscanner import ClusterFctType
from gnn_tracking.utils.device import guess_device
from gnn_tracking.utils.dictionaries import add_key_suffix
from gnn_tracking.utils.graph_masks import edge_subgraph, get_edge_mask_from_node_mask
from gnn_tracking.utils.log import get_logger
from gnn_tracking.utils.nomenclature import denote_pt
from gnn_tracking.utils.timing import timing

#: Function type that can be used as hook for the training/test step in the
#: `TCNTrainer` class. The function takes the trainer instance as first argument and
#: a dictionary of losses/metrics as second argument.
train_hook_type = Callable[["TCNTrainer", dict[str, Tensor]], None]
test_hook_type = Callable[["TCNTrainer", dict[str, Tensor]], None]
batch_hook_type = Callable[["TCNTrainer", int, int, dict[str, Tensor], Data], None]


@dataclass
class TrainingTruthCutConfig:
    """Configuration for truth cuts applied during training"""

    #: Truth cut on pt during training
    pt_thld: float = field(default=0.0)
    #: Remove noise hits during training
    without_noise: bool = field(default=False)
    #: Remove hits that are not reconstructable during training
    without_non_reconstructable: bool = field(default=False)

    def is_trivial(self) -> bool:
        """Return true if the truth cut is disabled"""
        return (
            np.isclose(self.pt_thld, 0.0)
            and not self.without_noise
            and not self.without_non_reconstructable
        )

    def get_masks(
        self,
        data: Data,
    ) -> tuple[Tensor, Tensor]:
        """Get mask for hits that are considered in training

        Returns:
            node mask, edge mask
        """
        node_mask = torch.full(
            (len(data.x),), True, dtype=torch.bool, device=data.x.device
        )
        if self.pt_thld > 0:
            # noise will also have pt = 0, so let's make sure we keep this independent
            no_noise_mask = data.particle_id > 0
            node_mask[no_noise_mask] &= data.pt[no_noise_mask] > self.pt_thld
        if self.without_noise:
            node_mask &= data.particle_id > 0
        if self.without_non_reconstructable:
            node_mask &= data.reconstructable > 0
        edge_mask = get_edge_mask_from_node_mask(
            node_mask=node_mask, edge_index=data.edge_index
        )
        return node_mask, edge_mask


# The following abbreviations are used throughout the code:
# W: edge weights
# B: condensation likelihoods
# H: clustering coordinates
# Y: edge truth labels
# L: hit truth labels
# P: Track parameters
class TCNTrainer:
    def __init__(
        self,
        model,
        loaders: dict[str, DataLoader] | dict[str, list[Data]],
        loss_functions: dict[str, LossFctType],
        *,
        device=None,
        lr: Any = 5e-4,
        optimizer: Callable = Adam,
        lr_scheduler: Callable | None = None,
        loss_weights: dict[str, float] | None = None,
        cluster_functions: dict[str, ClusterFctType] | None = None,
    ):
        """Main trainer class of the condensation network approach.

        Note: Additional (more advanced) settings are goverend by attributes rather
        than init arguments. Take a look at all attributes that do not start with
        ``_``.

        Args:
            model:
            loaders:
            loss_functions: Dictionary of loss functions, keyed by loss name
            device:
            lr: Learning rate
            optimizer: Optimizer to use (default: Adam): Function. Will be called with
                the model parameters as first positional parameter and with the learning
                rate as keyword argument (``lr``).
            lr_scheduler: Learning rate scheduler. If it needs parameters, apply
                ``functools.partial`` first
            loss_weights: Weight different loss functions.
                Either `DynamicLossWeights` object or a dictionary of weights keyed by
                loss name.
                If a dictionary and a key is left out, the weight is set to 1.0.
                The weights will be normalized to sum to 1.0 before use.
                If one of the loss functions called ``l`` returns a dictionary with keys
                k, the keys for loss_weights should be ``k_l``.
            cluster_functions: Dictionary of functions that take the output of the model
                during testing and report additional figures of merits (e.g.,
                clustering)
        """
        self.logger = get_logger("TCNTrainer", level=logging.INFO)
        self.device = guess_device(device)
        del device
        self.logger.info("Using device %s", self.device)
        #: Checkpoints are saved to this directory by default
        self.checkpoint_dir = Path(".")
        self.model = model.to(self.device)
        self.train_loader = loaders.get("train", None)
        self.test_loader = loaders.get("test", None)
        self.val_loader = loaders.get("val", None)

        self.loss_functions = {k: v.to(self.device) for k, v in loss_functions.items()}
        if cluster_functions is None:
            cluster_functions = {}
        self.clustering_functions = cluster_functions

        self._loss_weights = collections.defaultdict(lambda: 1.0)
        if loss_weights is not None:
            self._loss_weights.update(loss_weights)

        self.optimizer = optimizer(self.model.parameters(), lr=lr)
        self._lr_scheduler = lr_scheduler(self.optimizer) if lr_scheduler else None

        # Current epoch
        self._epoch = 0

        #: Hooks to be called after training epoch (please use `add_hook` to add them)
        self._train_hooks = list[train_hook_type]()
        #: Hooks to be called after testing (please use `add_hook` to add them)
        self._test_hooks = list[test_hook_type]()
        #: Hooks called after processing a batch (please use `add_hook` to add them)
        self._batch_hooks = list[batch_hook_type]()

        #: Mapping of cluster function name to best parameter
        self._best_cluster_params: dict[str, dict[str, Any] | None] = {}

        # output quantities
        self.train_loss = list[pd.DataFrame]()
        self.test_loss = list[pd.DataFrame]()

        #: Number of batches that are being used for the clustering functions and the
        #: evaluation of the related metrics.
        self.max_batches_for_clustering = 10

        self.training_truth_cuts = TrainingTruthCutConfig()

        #: pT thresholds that are being used in the evaluation of edge classification
        #: metrics in the test step
        self.ec_eval_pt_thlds = [0.9, 1.5]

        #: Do not run test step after training epoch
        self.skip_test_during_training = False

        # todo: This should rather be read from the model, because it makes only
        #   sense if it actually matches
        #: Threshold for edge classification in test step (does not
        #: affect training)
        self.ec_threshold = 0.5

    def add_hook(
        self, hook: train_hook_type | test_hook_type | batch_hook_type, called_at: str
    ) -> None:
        """Add a hook to training/test step

        Args:
            hook: Callable that takes a training model and a dictionary of tensors as
                inputs
            called_at: train or test

        Returns:
            None

        Example:


        """
        if called_at == "train":
            self._train_hooks.append(hook)
        elif called_at == "test":
            self._test_hooks.append(hook)
        elif called_at == "batch":
            self._batch_hooks.append(hook)
        else:
            raise ValueError("Invalid value for called_at")

    @staticmethod
    def _apply_mask(
        data: Data, node_mask: Tensor, edge_mask: Tensor | None = None
    ) -> Data:
        """Apply mask to data"""
        if edge_mask is not None:
            data = edge_subgraph(data, mask_to_index(edge_mask))
        node_index = mask_to_index(node_mask)
        data = data.subgraph(node_index)
        return data

    def evaluate_model(
        self, data: Data, mask_pids_reco=True, apply_truth_cuts=False
    ) -> dict[str, Tensor]:
        """Evaluate the model on the data and return a dictionary of outputs

        Args:
            data:
            mask_pids_reco: If True, mask out PIDs for non-reconstructables
            apply_truth_cuts: If True, apply pre-configured truth cuts (see
                `_apply_mask`)

        Returns:
            All values correspond to the truth cuts (if applied).
        """
        data = data.to(self.device)
        if apply_truth_cuts:
            node_mask, edge_mask = self.training_truth_cuts.get_masks(data)
            data = self._apply_mask(data, node_mask, edge_mask)

        out = self.model(data)

        if mask_pids_reco:
            pid_field = data.particle_id * data.reconstructable.long()
        else:
            pid_field = data.particle_id

        def squeeze_if_defined(key: str) -> None | Tensor:
            try:
                return out[key].squeeze()
            except KeyError:
                return None

        def get_if_defined(key: str) -> None | Tensor:
            try:
                return out[key]
            except KeyError:
                return None

        ec_hit_mask = out.get("ec_hit_mask", torch.full_like(data.pt, True)).bool()
        ec_edge_mask = out.get("ec_edge_mask", torch.full_like(data.y, True)).bool()

        dct = {
            # -------- flags
            "truth_cuts_applied": apply_truth_cuts,
            # -------- model_outputs
            "w": squeeze_if_defined("W"),
            "x": get_if_defined("H"),
            "beta": squeeze_if_defined("B"),
            "pred": get_if_defined("P"),
            "ec_hit_mask": ec_hit_mask,
            "ec_edge_mask": ec_edge_mask,
            # -------- from data
            "y": data.y,
            "particle_id": pid_field,
            # fixme: One of these is wrong
            "track_params": data.pt,
            "pt": data.pt,
            "reconstructable": data.reconstructable.long(),
            "edge_index": data.edge_index,
            "sector": data.sector,
            "node_features": data.x,
        }
        return dct

    def get_batch_losses(
        self, model_output: dict[str, Tensor]
    ) -> tuple[Tensor, dict[str, Tensor]]:
        """Calculate the losses for a batch of data

        Args:
            model_output:

        Returns:
            total loss, dictionary of losses, where total loss includes the weights
            assigned to the individual losses
        """
        individual_losses = {}
        for key, loss_func in self.loss_functions.items():
            loss = loss_func(**model_output)
            if isinstance(loss, Mapping):
                for k, v in loss.items():
                    individual_losses[f"{key}_{k}"] = v
            else:
                individual_losses[key] = loss

        total = sum(
            self._loss_weights[k] * individual_losses[k] for k in individual_losses
        )
        if torch.isnan(total):
            raise RuntimeError(
                f"NaN loss encountered in test step. {individual_losses=}."
            )
        return total, individual_losses

    def _log_losses(
        self,
        batch_losses: dict[str, Tensor | float],
        *,
        style="table",
        header: str = "",
    ) -> None:
        """Log the losses

        Args:
            batch_losses:
            style: "table" or "inline"
            header: Header to prepend to the log message

        Returns:
            None
        """
        report_str = header if header else ""
        if style == "table":
            report_str += "\n"
            non_error_keys: list[str] = sorted(
                [k for k in batch_losses if not k.endswith("_std")]
            )
            values = [batch_losses[k] for k in non_error_keys]
            errors = [batch_losses.get(f"{k}_std", "") for k in non_error_keys]
            markers = [
                "-->" if self.highlight_metric(key) else "" for key in non_error_keys
            ]
            annotated_table_items = zip(markers, non_error_keys, values, errors)
            report_str += tabulate.tabulate(
                annotated_table_items,
                tablefmt="outline",
                floatfmt=".5f",
                headers=["", "Metric", "Value", "Std"],
            )
        else:
            report_str += ", ".join(f"{k}={v:>10.5f}" for k, v in batch_losses.items())
        self.logger.info(report_str)

    def highlight_metric(self, metric: str) -> bool:
        """Should a metric be highlighted in the log output?"""
        metric = metric.casefold()
        if metric.startswith("tc_"):
            return False
        if "_loc_" in metric:
            return False
        if "0.9" not in metric and "1.5" not in metric:
            return False
        if "double_majority" in metric:
            return True
        if "tpr_eq_tnr" in metric:
            return True
        if "max_mcc" in metric:
            return True
        return False

    def train_step(self, *, max_batches: int | None = None) -> dict[str, float]:
        """

        Args:
            max_batches:  Only process this many batches per epoch (useful for testing
                to get to the validation step more quickly)

        Returns:
            Dictionary of losses
        """
        self.model.train()
        _losses = collections.defaultdict(list)
        n_oom_errors_in_a_row = 0
        assert self.train_loader is not None
        for batch_idx, data in enumerate(self.train_loader):
            if max_batches and batch_idx > max_batches:
                break
            try:
                data = data.to(self.device)
                model_output = self.evaluate_model(data, apply_truth_cuts=True)
                batch_loss, batch_losses = self.get_batch_losses(model_output)
                self.optimizer.zero_grad(set_to_none=True)
                batch_loss.backward()
                self.optimizer.step()
            except RuntimeError as e:
                if "out of memory" in str(e).casefold():
                    n_oom_errors_in_a_row += 1
                    self.logger.warning(
                        "WARNING: ran out of memory (OOM), skipping batch. "
                        "If this happens frequently, decrease the batch size."
                        "Will abort if we get 10 consecutive OOM errors."
                    )
                    if n_oom_errors_in_a_row > 10:
                        raise
                    continue
                raise
            else:
                n_oom_errors_in_a_row = 0

            for hook in self._batch_hooks:
                hook(self, self._epoch, batch_idx, model_output, data)

            if (batch_idx % 10) == 0:
                _losses_w = {}
                for key, loss in batch_losses.items():
                    _losses_w[f"{key}_weighted"] = loss.item() * self._loss_weights[key]
                self._log_losses(
                    # batch_losses,
                    _losses_w,
                    header=f"Epoch {self._epoch:>2} "
                    f"({batch_idx:>5}/{len(self.train_loader)}): ",
                    style="inline",
                )

            _losses["total"].append(batch_loss.item())
            for key, loss in batch_losses.items():
                _losses[f"{key}"].append(loss.item())
                _losses[f"{key}_weighted"].append(loss.item() * self._loss_weights[key])

        losses = {k: np.nanmean(v) for k, v in _losses.items()}
        self.train_loss.append(pd.DataFrame(losses, index=[self._epoch]))
        for hook in self._train_hooks:
            hook(self, losses)
        return losses

    @staticmethod
    def _edge_pt_mask(edge_index: Tensor, pt: Tensor, pt_min=0.0) -> Tensor:
        """Mask edges where BOTH (!) nodes have pt <= pt_min.
        Note how the resulting edge mask is different from the pt truth cut mask,
        which only requires one of the nodes of the edge to have pt <= pt_min to
        be masked.
        """
        pt_a = pt[edge_index[0]]
        pt_b = pt[edge_index[1]]
        return (pt_a > pt_min) | (pt_b > pt_min)

    @torch.no_grad()
    def single_test_step(
        self, val=True, apply_truth_cuts=False, max_batches: int | None = None
    ) -> dict[str, float]:
        """Test the model on the validation or test set

        Args:
            val: Use validation dataset rather than test dataset
            apply_truth_cuts: Apply truth cuts (e.g., truth level pt cut) during
                the evaluation
            max_batches: Only process this many batches per epoch (useful for testing)

        Returns:
            Dictionary of metrics
        """
        self.model.eval()

        # We connect part of the data in CPU memory for clustering & evaluation
        cluster_eval_input: DefaultDict[
            str, list[np.ndarray]
        ] = collections.defaultdict(list)

        batch_metrics = collections.defaultdict(list)
        loader = self.val_loader if val else self.test_loader
        assert loader is not None
        for _batch_idx, data in enumerate(loader):
            if max_batches and _batch_idx > max_batches:
                break
            data = data.to(self.device)
            model_output = self.evaluate_model(
                data, mask_pids_reco=False, apply_truth_cuts=apply_truth_cuts
            )
            batch_loss, these_batch_losses = self.get_batch_losses(model_output)

            batch_metrics["total"].append(batch_loss.item())
            for key, value in these_batch_losses.items():
                batch_metrics[key].append(value.item())
                batch_metrics[f"{key}_weighted"].append(
                    value.item() * self._loss_weights[key]
                )

            for key, value in self.evaluate_ec_metrics(
                model_output,
            ).items():
                batch_metrics[key].append(value)

            # Build up a dictionary of inputs for clustering (note that we need to
            # map the names of the model outputs to the names of the clustering input)
            if (
                self.clustering_functions
                and _batch_idx <= self.max_batches_for_clustering
            ):
                for mo_key, cf_key in ClusterFctType.required_model_outputs.items():
                    cluster_eval_input[cf_key].append(
                        model_output[mo_key].detach().cpu().numpy()
                    )

        # Merge all metrics in one big dictionary
        metrics: dict[str, float] = (
            {k: np.nanmean(v) for k, v in batch_metrics.items()}
            | {
                f"{k}_std": np.nanstd(v, ddof=1).item()
                for k, v in batch_metrics.items()
            }
            | self._evaluate_cluster_metrics(cluster_eval_input)
        )

        self.test_loss.append(pd.DataFrame(metrics, index=[self._epoch]))
        for hook in self._test_hooks:
            hook(self, metrics)
        return metrics

    def _evaluate_cluster_metrics(
        self, cluster_eval_input: dict[str, list[np.ndarray]]
    ) -> dict[str, float]:
        """Perform cluster studies and evaluate corresponding metrics

        Args:
            cluster_eval_input: Dictionary of inputs for clustering collected in
                `single_test_step`

        Returns:
            Dictionary of cluster metrics
        """
        metrics = {}
        for fct_name, fct in self.clustering_functions.items():
            cluster_result = fct(
                **cluster_eval_input,
                epoch=self._epoch,
                start_params=self._best_cluster_params.get(fct_name),
            )
            if cluster_result is None:
                continue
            metrics.update(cluster_result.metrics)
            self._best_cluster_params[fct_name] = cluster_result.best_params
            metrics.update(
                {
                    f"best_{fct_name}_{param}": val
                    for param, val in cluster_result.best_params.items()
                }
            )
        return metrics

    @torch.no_grad()
    def evaluate_ec_metrics_with_pt_thld(
        self, model_output: dict[str, torch.Tensor], pt_min: float, ec_threshold: float
    ) -> dict[str, float]:
        """Evaluate edge classification metrics for a given pt threshold and
        EC threshold.

        Args:
            model_output: Output of the model
            pt_min: pt threshold: We discard all edges where both nodes have
                `pt <= pt_min` before evaluating any metric.
            ec_threshold: EC threshold

        Returns:
            Dictionary of metrics
        """
        edge_pt_mask = self._edge_pt_mask(
            model_output["edge_index"], model_output["pt"], pt_min
        )
        predicted = model_output["w"][edge_pt_mask]
        true = model_output["y"][edge_pt_mask].long()

        bcs = BinaryClassificationStats(
            output=predicted,
            y=true,
            thld=ec_threshold,
        )
        metrics = bcs.get_all() | get_maximized_bcs(output=predicted, y=true)
        metrics["roc_auc"] = roc_auc_score(y_true=true, y_score=predicted)
        for max_fpr in [
            0.001,
            0.01,
            0.1,
        ]:
            metrics[f"roc_auc_{max_fpr}FPR"] = roc_auc_score(
                y_true=true,
                y_score=predicted,
                max_fpr=max_fpr,
            )
        return {denote_pt(k, pt_min): v for k, v in metrics.items()}

    @torch.no_grad()
    def evaluate_ec_metrics(
        self, model_output: dict[str, torch.Tensor], ec_threshold: float | None = None
    ) -> dict[str, float]:
        """Evaluate edge classification metrics for all pt thresholds."""
        if ec_threshold is None:
            ec_threshold = self.ec_threshold
        if model_output["w"] is None:
            return {}
        ret = {}
        for pt_min in self.ec_eval_pt_thlds:
            ret.update(
                self.evaluate_ec_metrics_with_pt_thld(
                    model_output, pt_min, ec_threshold=ec_threshold
                )
            )
        return ret

    def test_step(self, val=True, max_batches: int | None = None) -> dict[str, float]:
        """Validate the model and test the model on the validation/test set.
        This method is called during training and makes multiple calls to
        `single_test_step` corresponding to truth cut or uncut data.

        Args:
            val: Use validation dataset rather than test dataset
            max_batches: Use a maximum number of batches for testing
        """
        test_results = self.single_test_step(val=val, max_batches=max_batches)
        if not self.training_truth_cuts.is_trivial():
            test_results.update(
                add_key_suffix(
                    self.single_test_step(
                        val=val,
                        apply_truth_cuts=True,
                    ),
                    "tc_",
                ),
            )
        return test_results

    def step(self, *, max_batches: int | None = None) -> dict[str, float]:
        """Train one epoch and test

        Args:
            max_batches: See train_step
        """
        self._epoch += 1
        with timing(f"Training for epoch {self._epoch}", self.logger):
            train_losses = self.train_step(max_batches=max_batches)
        if not self.skip_test_during_training:
            with timing(f"Test step for epoch {self._epoch}", self.logger):
                test_results = self.test_step(max_batches=max_batches)
        else:
            test_results = {}
        results = {
            **{f"{k}_train": v for k, v in train_losses.items()},
            **test_results,
        }
        self._log_losses(
            results,
            header=f"Results {self._epoch}: ",
        )
        if self._lr_scheduler:
            self._lr_scheduler.step()
        return results

    def train(self, epochs=1000, max_batches: int | None = None):
        """Train the model.

        Args:
            epochs:
            max_batches: See train_step.

        Returns:

        """
        for _ in range(1, epochs + 1):
            try:
                self.step(max_batches=max_batches)
            except KeyboardInterrupt:
                self.logger.warning("Keyboard interrupt")
                self.save_checkpoint()
                raise
        self.save_checkpoint()

    # noinspection PyMethodMayBeStatic
    def get_checkpoint_name(self) -> str:
        """Generate name of checkpoint file based on current time."""
        now = datetime.now()
        return f"{now:%y%m%d_%H%M%S}_model.pt"

    def get_checkpoint_path(self, path: str | PurePath = "") -> Path:
        """Get checkpoint path based on user input."""
        if not path:
            return self.checkpoint_dir / self.get_checkpoint_name()
        if isinstance(path, str) and os.sep not in path:
            return self.checkpoint_dir / path
        return Path(path)

    def save_checkpoint(self, path: str | PurePath = "") -> None:
        """Save state of model, optimizer and more for later resuming of training."""
        path = self.get_checkpoint_path(path)
        self.logger.info("Saving checkpoint to %s", path)
        torch.save(
            {
                "epoch": self._epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
            },
            path,
        )

    def load_checkpoint(self, path: str | PurePath, device=None) -> None:
        """Resume training from checkpoint"""
        device = guess_device(device)
        checkpoint = torch.load(self.get_checkpoint_path(path), map_location=device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self._epoch = checkpoint["epoch"]
