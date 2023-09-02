import os
from pathlib import Path

import torch
import yaml
from pytorch_lightning.core.mixins import HyperparametersMixin
from torch import nn
from torch_geometric.data import Data
from tqdm import tqdm

from gnn_tracking.models.edge_classifier import ECFromChkpt
from gnn_tracking.utils.log import logger


class DataTransformer:
    def __init__(
        self,
        transform: nn.Module,
    ):
        """Applies a transformation function to all data files and saves
        them on disk.
        """
        self._transform = transform

    def process(
        self,
        input_dir: os.PathLike,
        output_dir: os.PathLike,
        filename: str,
        *,
        redo=False,
    ) -> None:
        """Process single file"""
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        in_path = input_dir / filename
        if not redo and in_path.exists():
            logger.debug("File %s already exists, skipping", in_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        out_path = output_dir / filename
        data = torch.load(in_path)
        transformed = self._transform(data)
        torch.save(transformed, out_path)

    def _save_hparams(
        self,
        input_dir: Path,
        output_dir: Path,
    ) -> None:
        """Save hyperparameters to disk"""
        output_dir.mkdir(parents=True, exist_ok=True)
        hparams = dict(self._transform.hparams)
        hparams["input_dir"] = input_dir
        (output_dir / "hparams").write_text(yaml.dump(hparams))

    def process_directories(
        self,
        input_dirs: list[os.PathLike],
        output_dirs: list[os.PathLike],
        *,
        redo=True,
        progress=True,
        _first_only=False,
    ) -> None:
        """Process all files in the input directories and save them to the output
        directories.

        Args:
            input_dirs:
            output_dirs:
            redo: If True, overwrite existing files
            progress: Show progress bar
            _first_only: Only process the first file. Useful for testing.

        Returns:
            None
        """
        input_dirs = [Path(p) for p in input_dirs]
        output_dirs = [Path(p) for p in output_dirs]
        if len(input_dirs) != len(output_dirs):
            msg = "input_dirs and output_dirs must have the same length"
            raise ValueError(msg)
        directories = list(zip(input_dirs, output_dirs))
        for input_dir, output_dir in directories:
            self._save_hparams(input_dir, output_dir)
            input_filenames = [p.name for p in input_dir.glob("*.pt")]
            iterator = input_filenames
            if progress:
                iterator = tqdm(iterator)
            for filename in iterator:
                self.process(input_dir, output_dir, filename, redo=redo)
                if _first_only:
                    break


class ECCut(nn.Module, HyperparametersMixin):
    # noinspection PyUnusedLocal
    def __init__(
        self,
        chkpt_path: str,
        thld: float,  # noqa: ARG002
        *,
        class_name="gnn_tracking.training.ec.ECModule",
        device: str | None = None,
    ):
        """Applies a cut to the edge classifier output and saves the trimmed down
        graphs.

        Args:
        """
        super().__init__()
        self.save_hyperparameters()
        self._model = ECFromChkpt(chkpt_path, class_name=class_name, device=device)

    def __forward__(self, data) -> Data:
        w = self._model(data)["W"]
        mask = w > self.hparams.thld
        data.ec_score = w
        return data.edge_subgraph(mask)
