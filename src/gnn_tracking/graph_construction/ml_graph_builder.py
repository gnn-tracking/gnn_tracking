from pathlib import Path

import torch
import yaml
from torch import nn
from tqdm import tqdm


class MLGraphBuilder:
    def __init__(
        self,
        gc: nn.Module,
    ):
        """Applies a metric learning graph construction to all point clouds and saves
        them on disk.

        Args:

        """
        self._gc = gc

    def process(self, input_dir: Path, output_dir: Path, filename: str) -> None:
        """Process single file"""
        in_path = input_dir / filename
        output_dir.mkdir(parents=True, exist_ok=True)
        out_path = output_dir / filename
        data = torch.load(in_path)
        transformed = self._gc(data)
        torch.save(transformed, out_path)

    def _save_hparams(self, input_dir: Path, output_dir: Path) -> None:
        """Save hyperparameters to disk"""
        output_dir.mkdir(parents=True, exist_ok=True)
        hparams = dict(self._gc.hparams)
        hparams["input_dir"] = input_dir
        (output_dir / "hparams").write_text(yaml.dump(hparams))

    def process_directories(
        self,
        input_dirs: list[Path],
        output_dirs: list[Path],
        *,
        progress=True,
        _first_only=False
    ) -> None:
        """Process all files in the input directories and save them to the output
        directories.

        Args:
            input_dirs:
            output_dirs:
            progress: Show progress bar
            _first_only: Only process the first file. Useful for testing.

        Returns:
            None
        """
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
                self.process(input_dir, output_dir, filename)
                if _first_only:
                    break
