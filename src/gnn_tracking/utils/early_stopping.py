from __future__ import annotations

from abc import ABC, abstractmethod

# Loosely following the implementation from
# https://debuggercafe.com/using-learning-rate-scheduler-and-early-stopping-with-pytorch/
import numpy as np


class StopEarly(ABC):
    @abstractmethod
    def __call__(self, loss: float) -> bool:
        """Returns True if training should be stopped."""
        pass


class DontStopEarly(StopEarly):
    def __call__(self, loss: float) -> bool:
        return False


dont_stop_early = DontStopEarly()


class StopEarlyRatio(StopEarly):
    def __init__(self, patience=5, ratio_threshold=1, running=2):
        """Stop training if the loss does not improve for a given number of epochs,
        where a certain ratio has to be surpassed for an epoch to be considered an
        improvement.

        Args:
            patience: Number of epochs of no improvement after which training will be
                stopped.
            ratio_threshold: Minimum ratio in loss that we would consider an
                improvement.
            running: For the current best loss consider the mean of this many epochs.
                If set to 1, no running average is used.

        """
        assert patience > 0
        self.patience = patience
        assert ratio_threshold < 1
        self.ratio_threshold = ratio_threshold
        self._best_loss = None
        self._times_no_improvement = 0
        self._losses = []
        self.running = running

    def __call__(self, loss: float) -> bool:
        """Returns True if training should be stopped."""
        self._losses.append(loss)
        del loss
        if len(self._losses) < self.running:
            return False
        running_loss = np.mean(self._losses[-self.running :])
        if self._best_loss is None:
            self._best_loss = running_loss
            return False
        elif self._best_loss / running_loss > self.ratio_threshold:
            self._best_loss = running_loss
            self._times_no_improvement = 0
            return False
        else:
            self._times_no_improvement += 1
            print(
                f"INFO: Early stopping counter: "
                f"{self._times_no_improvement}/{self.patience}"
            )
            if self._times_no_improvement >= self.patience:
                print("INFO: Early stopping")
                return True
