import importlib
from typing import Any

from pytorch_lightning import LightningModule
from pytorch_lightning.core.mixins import HyperparametersMixin

from gnn_tracking.utils.log import logger


def save_sub_hyperparameters(
    self: HyperparametersMixin, key: str, obj: HyperparametersMixin | dict
) -> None:
    """Take hyperparameters from `obj` and save them to `self` under the
    key `key`.
    """
    assert key not in self.hparams
    if isinstance(obj, dict):
        hparams = obj
    else:
        hparams = dict(obj.hparams)
    sub_hparams = {
        "class_path": obj.__class__.__module__ + "." + obj.__class__.__name__,
        "init_args": hparams,
    }
    self.save_hyperparameters({key: sub_hparams})


def get_object_from_path(path: str) -> Any:
    """Get object from path (string) to its code location."""
    module_name, _, class_name = path.rpartition(".")
    logger.debug("Getting class %s from module %s", class_name, module_name)
    if not module_name:
        raise ValueError("Please specify the full import path")
    module = importlib.import_module(module_name)
    return getattr(module, class_name)


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
    model = model_class.load_from_checkpoint(chkpt_path)
    logger.debug("Checkpoint loaded. Model ready to go.")
    return model
