from pathlib import Path

import pytest
from pytorch_lightning.callbacks import RichProgressBar
from pytorch_lightning.cli import LightningCLI
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger

from gnn_tracking.training.callbacks import ExpandWandbConfig, PrintValidationMetrics
from gnn_tracking.utils.loading import TrackingDataModule
from tests.test_configs import test_config_dir
from tests.test_data import test_data_dir

AVAILABLE_CONFIGS: list[Path] = list(test_config_dir.glob("*.yml"))


class TrackingDataModuleForTests(TrackingDataModule):
    def __new__(cls, *args, **kwargs) -> TrackingDataModule:
        return TrackingDataModule(
            identifier="test",
            train={"dirs": [test_data_dir / "graphs"]},
            val={"dirs": [test_data_dir / "graphs"]},
        )


@pytest.mark.slow()
@pytest.mark.parametrize("config_file", AVAILABLE_CONFIGS)
def test_train_from_config(config_file: Path, tmp_path):
    logger = WandbLogger(
        project="test",
        group="test",
        offline=True,
        version="test",
        save_dir=tmp_path,
    )

    tb_logger = TensorBoardLogger(tmp_path, version="test")

    cli = LightningCLI(  # noqa: F841
        datamodule_class=TrackingDataModuleForTests,
        trainer_defaults={
            "callbacks": [
                RichProgressBar(leave=True),
                PrintValidationMetrics(),
                ExpandWandbConfig(),
            ],
            "log_every_n_steps": 1,
            "accelerator": "cpu",
            "max_steps": 1,
            "num_sanity_val_steps": 0,
            "logger": [tb_logger, logger],
        },
        args=["fit", "--config", str(config_file)],
    )
