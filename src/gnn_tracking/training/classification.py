import torch
# import torchmetrics
from typing import Any

from torch import Tensor
from torch import Tensor as T
from torch_geometric.data import Data

from torch.nn import CrossEntropyLoss, BCELoss
from gnn_tracking.metrics.noise_classification import get_fp_pt
from gnn_tracking.training.base import TrackingModule
from gnn_tracking.utils.dictionaries import add_key_suffix, to_floats
from gnn_tracking.utils.lightning import obj_from_or_to_hparams
from gnn_tracking.utils.oom import tolerate_some_oom_errors


class NodeClassifierModule(TrackingModule):
    # noinspection PyUnusedLocal
    def __init__(
        self,
        *,
        loss_fct: torch.nn,
        **kwargs,
    ):
        """Pytorch lightning module with training and validation step for the metric
        learning approach to graph construction.
        """
        super().__init__(**kwargs)
        self.loss_fct: BCELoss | CrossEntropyLoss = obj_from_or_to_hparams(
            self, "loss_fct", loss_fct
        )
        # self.valid_acc = torchmetrics.Accuracy(task="binary")
        # self.roc = torchmetrics.ROC(task='binary')

    # noinspection PyUnusedLocal
    def get_losses(self, out: dict[str, Any], data: Data) -> tuple[T, dict[str, float]]:
        # targets = torch.vstack([data.particle_id == 0, data.particle_id != 0]).type(torch.LongTensor).to('cuda')
        targets = torch.tensor(list(zip(data.particle_id == 0, data.particle_id != 0))).type(torch.FloatTensor).to('cuda')
        loss = self.loss_fct(out["H"], targets)
        metrics = {}
        metrics["total"] = float(loss)
        return loss, metrics

    @tolerate_some_oom_errors
    def training_step(self, batch: Data, batch_idx: int) -> Tensor | None:
        batch = self.data_preproc(batch)
        out = self(batch, _preprocessed=True)
        loss, loss_dct = self.get_losses(out, batch)
        self.log_dict(
            add_key_suffix(loss_dct, "_train"),
            prog_bar=True,
            on_step=True,
            batch_size=self.trainer.train_dataloader.batch_size,
        )
        return loss

    def validation_step(self, batch: Data, batch_idx: int):
        batch = self.data_preproc(batch)
        out = self(batch, _preprocessed=True)
        loss, loss_dct = self.get_losses(out, batch)
        self.log_dict(
            add_key_suffix(loss_dct, "_val"), 
            prog_bar=True,
            on_step=True,
            batch_size=self.trainer.val_dataloaders.batch_size
        )
        metrics = {}
        metrics["fp_pt"] = get_fp_pt(batch, self.model)
        self.log_dict(
            metrics,
            batch_size=self.trainer.val_dataloaders.batch_size
        )

    def on_validation_epoch_end(self) -> None:
        pass

    def highlight_metric(self, metric: str) -> bool:
        return metric in [
            "n_edges_frac_segment50_95",
            "total",
            "attractive",
            "repulsive",
            "max_frac_segment50",
        ]