from pytorch_lightning.core.mixins import HyperparametersMixin


def save_sub_hyperparameters(
    self: HyperparametersMixin, key: str, obj: HyperparametersMixin | dict
):
    assert key not in self.hparams
    sub_hparams = {}
    sub_hparams["class_path"] = obj.__class__.__module__ + "." + obj.__class__.__name__
    if isinstance(obj, dict):
        hparams = obj
    else:
        hparams = dict(obj.hparams)
    sub_hparams["init_args"] = hparams
    self.save_hyperparameters({key: sub_hparams})
