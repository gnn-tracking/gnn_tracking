import importlib
import math
import os
from pathlib import Path
from typing import Any, Literal

import pytorch_lightning
import torch
from pytorch_lightning import LightningModule
from pytorch_lightning.core.mixins.hparams_mixin import HyperparametersMixin
from torch import Tensor, nn
from torchmetrics import Metric
from tqdm import tqdm

from gnn_tracking.utils.log import logger


def save_sub_hyperparameters(
    self: HyperparametersMixin,
    key: str,
    obj: HyperparametersMixin | dict,
    errors: Literal["warn", "raise"] = "warn",
) -> None:
    """Take hyperparameters from `obj` and save them to `self` under the
    key `key`.

    Args:
        self: The object to save the hyperparameters to.
        key: The key under which to save the hyperparameters.
        obj: The object to take the hyperparameters from.
        errors: Whether to raise an error or just warn
    """
    if not hasattr(obj, "hparams"):
        msg = (
            "Can't save hyperparameters from object of type %s. Make sure to "
            "inherit from HyperparametersMixin."
        )
        if errors == "warn":
            logger.warning(msg, type(obj))
            return
        if errors == "raise":
            _ = msg % type(obj)
            raise ValueError(_)
        _ = f"Unknown value for `errors`: {errors}"
        raise ValueError(_)

    assert key not in self.hparams
    if isinstance(obj, dict):
        logger.warning("SSH got dict %s. That's unexpected.", obj)
        sub_hparams = obj
    else:
        sub_hparams = {
            "class_path": obj.__class__.__module__ + "." + obj.__class__.__name__,
            "init_args": dict(obj.hparams),
        }
    self.save_hyperparameters({key: sub_hparams})


def load_obj_from_hparams(hparams: dict[str, Any], key: str = "") -> Any:
    """Load object from hyperparameters."""
    if key:
        hparams = hparams[key]
    return get_object_from_path(hparams["class_path"], hparams["init_args"])


def obj_from_or_to_hparams(self: HyperparametersMixin, key: str, obj: Any) -> Any:
    """Used to support initializing python objects from hyperparameters:
    If `obj` is a python object other than a dictionary, its hyperparameters are
    saved (its class path and init args) to `self.hparams[key]`.
    If `obj` is instead a dictionary, its assumed that we have to restore an object
    based on this information.
    """
    if isinstance(obj, dict) and "class_path" in obj and "init_args" in obj:
        self.save_hyperparameters({key: obj})
        return load_obj_from_hparams(obj)
    if isinstance(obj, (int, float, str, bool, list, tuple, dict)) or obj is None:
        self.save_hyperparameters({key: obj})
        return obj
    save_sub_hyperparameters(self=self, key=key, obj=obj)  # type: ignore
    return obj


def get_object_from_path(path: str, init_args: dict[str, Any] | None = None) -> Any:
    """Get object from path (string) to its code location."""
    module_name, _, class_name = path.rpartition(".")
    logger.debug("Getting class %s from module %s", class_name, module_name)
    if not module_name:
        msg = "Please specify the full import path"
        raise ValueError(msg)
    module = importlib.import_module(module_name)
    obj = getattr(module, class_name)
    if init_args is not None:
        return obj(**init_args)
    return obj


def get_lightning_module(
    class_path: str,
    chkpt_path: str = "",
    *,
    freeze: bool = True,
    device: str | None = None,
) -> LightningModule | None:
    """Get model (specified by `class_path`, a string) and
    load a checkpoint.
    """
    if class_path is None:
        return None
    if not chkpt_path:
        msg = (
            "This function currently does not support to restore a model "
            "without specifying a Checkpoint."
        )
        raise ValueError(msg)
    model_class: type = get_object_from_path(class_path)
    assert issubclass(model_class, LightningModule)
    logger.debug("Loading checkpoint %s", chkpt_path)
    model = model_class.load_from_checkpoint(
        chkpt_path, strict=False, map_location=device
    )
    model.hparams["restored_chkpt_path"] = chkpt_path
    logger.debug("Checkpoint loaded. Model ready to go.")
    if freeze:
        model.freeze()
    return model


def get_model(
    class_path: str,
    chkpt_path: str = "",
    freeze: bool = False,
    whole_module: bool = False,
    device: None | str = None,
) -> nn.Module | None:
    """Get torch model (specified by `class_path`, a string) and load a checkpoint.
    Uses `get_lightning_module` to get the model.

    Args:
        class_path: The path to the lightning module class
        chkpt_path: The path to the checkpoint. If no checkpoint is specified, we
            return None.
        freeze: Whether to freeze the model
        whole_module: Whether to return the whole lightning module or just the model
        device:
    """
    if not chkpt_path:
        return None
    lm = get_lightning_module(class_path, chkpt_path, freeze=freeze, device=device)
    if lm is None:
        return None
    if whole_module:
        return lm
    return lm.model


class StandardError(Metric):
    def __init__(
        self,
    ):
        """A torch metric that computes the standard error.
        This is necessary, because LightningModule.log doesn't take custom
        reduce functions.
        """
        super().__init__()
        self.add_state("values", default=torch.tensor([]), dist_reduce_fx="cat")

    def update(self, x: Tensor):
        self.values = torch.cat((self.values, x))

    def compute(self):
        return torch.std(self.values) / math.sqrt(len(self.values))


class SimpleTqdmProgressBar(pytorch_lightning.callbacks.ProgressBar):
    def __init__(self):
        """Fallback progress bar that creates a new tqdm bar for each epoch.
        Adapted from https://github.com/Lightning-AI/lightning/issues/2189 , reply
        https://github.com/Lightning-AI/lightning/issues/2189#issuecomment-1510439811
        """
        super().__init__()
        self.bar = None
        self.enabled = True

    @property
    def is_enabled(self):
        return self.enabled

    def on_train_epoch_start(self, trainer, pl_module):  # noqa: ARG002
        if self.enabled:
            self.bar = tqdm(
                total=self.total_train_batches,
                desc=f"Epoch {trainer.current_epoch+1}",
                position=0,
                leave=True,
            )

    def on_train_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx  # noqa: ARG002
    ):
        if self.bar:
            self.bar.update(1)
            self.bar.set_postfix(self.get_metrics(trainer, pl_module))

    def on_validation_epoch_end(self, trainer, pl_module) -> None:
        if self.bar:
            self.bar.set_postfix(self.get_metrics(trainer, pl_module))
            self.bar.close()
            self.bar = None

    def disable(self):
        self.bar = None
        self.enabled = False


def find_latest_checkpoint(
    log_dir: os.PathLike,
    trial_name: str = "",
) -> Path:
    """Find latest lightning checkpoint

    Args:
        log_dir (os.PathLike): Path to the directory of your trial or to the
            directory of the experiment if `trial_name` is specified.
        trial_name (str, optional): Name of the trial if `log_dir` is the
            directory of the experiment.
    """
    log_dir = Path(log_dir)
    if trial_name:
        log_dir /= trial_name
    assert log_dir.is_dir()
    if log_dir.name != "checkpoints":
        log_dir /= "checkpoints"
    assert log_dir.is_dir()
    checkpoints = list(log_dir.glob("*.ckpt"))
    if len(checkpoints) == 0:
        msg = f"No checkpoints found in {log_dir}"
        raise ValueError(msg)
    return max(checkpoints, key=lambda p: p.stat().st_mtime)
