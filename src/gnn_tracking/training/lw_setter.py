from __future__ import annotations

from typing import Any

import numpy as np
from torch_geometric.data import Data

from gnn_tracking.utils.log import logger


class LossWeightSetterHook:
    def __call__(
        self,
        trainer,
        epoch: int,
        batch_idx: int,
        model_output: dict[str, Any],
        data: Data,
    ):
        ...


class SineLWSH(LossWeightSetterHook):
    def __init__(
        self,
        loss_name: str | tuple[str, ...],
        mean: float,
        amplitude: float,
        period: int,
        amplitude_halflife: float,
        n_batches: int,
    ):
        self._loss_name = loss_name
        self._mean = mean
        self._amplitude = amplitude
        self._period = period
        self._amplitude_half_life = amplitude_halflife
        self._n_batches = n_batches

    def get_lw(self, epoch, batch_idx):
        idx = epoch * self._n_batches + batch_idx
        amplitude_decay = 0.5 ** (1 / (self._amplitude_half_life * self._n_batches))
        amplitude = self._amplitude * amplitude_decay**idx
        s = np.sin(2 * np.pi * idx / (self._period * self._n_batches))
        r = self._mean + amplitude * s
        logger.debug("Setting loss weight to %f", r)
        return r

    def __call__(
        self,
        trainer,
        epoch: int,
        batch_idx: int,
        model_output: dict[str, Any],
        data: Data,
    ):
        if isinstance(self._loss_name, str):
            assert self._loss_name in trainer.loss_functions
            trainer.loss_functions[self._loss_name][1] = self.get_lw(epoch, batch_idx)
        else:
            assert len(self._loss_name) == 2
            assert self._loss_name[1] in trainer.loss_functions[self._loss_name[0]][1]
            trainer.loss_functions[self._loss_name[0]][1][
                self._loss_name[1]
            ] = self.get_lw(epoch, batch_idx)
