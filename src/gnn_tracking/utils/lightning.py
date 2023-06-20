import importlib
import math
from typing import Any

import torch
from pytorch_lightning import LightningModule
from pytorch_lightning.core.mixins import HyperparametersMixin
from torch import Tensor
from torchmetrics import Metric

from gnn_tracking.utils.log import logger


def save_sub_hyperparameters(
    self: HyperparametersMixin, key: str, obj: HyperparametersMixin | dict
) -> None:
    """Take hyperparameters from `obj` and save them to `self` under the
    key `key`.
    """
    assert key not in self.hparams
    if isinstance(obj, dict):
        logger.warning("SSH got dict %s. That's unexpected.", obj)
        sub_hparams = obj
    else:
        sub_hparams = {
            "class_path": obj.__class__.__module__ + "." + obj.__class__.__name__,
            "init_args": dict(obj.hparams),
        }
    logger.debug("Saving hyperperameters %s", sub_hparams)
    self.save_hyperparameters({key: sub_hparams})


def load_obj_from_hparams(hparams: dict[str, Any], key: str = "") -> Any:
    """Load object from hyperparameters."""
    if key:
        hparams = hparams[key]
    return get_object_from_path(hparams["class_path"], hparams["init_args"])


def obj_from_or_to_hparams(self: HyperparametersMixin, key: str, obj: Any) -> Any:
    if obj is None:
        return None
    if isinstance(obj, dict):
        logger.debug("Got %s, assuming I have to load", obj)
        # Assume that we have to load
        return load_obj_from_hparams(obj)
    logger.debug(
        "Got obj of type %s, assuming I have to save hyperparameters", type(obj)
    )
    save_sub_hyperparameters(self=self, key=key, obj=obj)  # type: ignore
    return obj


def get_object_from_path(path: str, init_args: dict[str, Any] | None = None) -> Any:
    """Get object from path (string) to its code location."""
    module_name, _, class_name = path.rpartition(".")
    logger.debug("Getting class %s from module %s", class_name, module_name)
    if not module_name:
        raise ValueError("Please specify the full import path")
    module = importlib.import_module(module_name)
    obj = getattr(module, class_name)
    if init_args is not None:
        return obj(**init_args)
    return obj


def get_model(class_path: str, chkpt_path: str = "") -> LightningModule | None:
    """Get model (specified by `class_path`, a string) and
    load a checkpoint.
    """
    if class_path is None:
        return None
    if not chkpt_path:
        raise ValueError(
            "This function currently does not support to restore a model without "
            "specifying a Checkpoint."
        )
    model_class: type = get_object_from_path(class_path)
    assert issubclass(model_class, LightningModule)
    logger.debug("Loading checkpoint %s", chkpt_path)
    model = model_class.load_from_checkpoint(chkpt_path, strict=False)
    logger.debug("Checkpoint loaded. Model ready to go.")
    return model


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
