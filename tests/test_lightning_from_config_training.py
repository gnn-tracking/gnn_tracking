from pathlib import Path

import pytest
from pytorch_lightning.callbacks import RichProgressBar
from pytorch_lightning.cli import LightningCLI

from gnn_tracking.training.callbacks import PrintValidationMetrics
from gnn_tracking.utils.loading import TrackingDataModule

from .test_configs import test_config_dir
from .test_data import test_data_dir

AVAILABLE_CONFIGS: list[Path] = list(test_config_dir.glob("*.yml"))


def tracking_data_module() -> TrackingDataModule:
    return TrackingDataModule(
        train={"dirs": [test_data_dir / "graphs"]},
        val={"dirs": [test_data_dir / "graphs"]},
    )


@pytest.mark.parametrize("config_file", AVAILABLE_CONFIGS)
def test_train_from_config(config_file: Path):
    cli = LightningCLI(  # noqa F841
        datamodule_class=tracking_data_module,
        trainer_defaults=dict(
            callbacks=[
                RichProgressBar(leave=True),
                PrintValidationMetrics(),
            ],
            log_every_n_steps=1,
            accelerator="cpu",
            max_epochs=1,
        ),
        args=["fit", "--config", str(config_file)],
    )
