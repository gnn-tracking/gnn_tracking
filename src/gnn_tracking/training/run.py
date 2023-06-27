from pytorch_lightning.callbacks import RichProgressBar
from pytorch_lightning.cli import LightningCLI

from gnn_tracking.utils.loading import TrackingDataModule


def cli_main():
    # noinspection PyUnusedLocal
    cli = LightningCLI(  # noqa F841
        datamodule_class=TrackingDataModule,
        trainer_defaults=dict(callbacks=[RichProgressBar(leave=True)]),
    )


if __name__ == "__main__":
    cli_main()
