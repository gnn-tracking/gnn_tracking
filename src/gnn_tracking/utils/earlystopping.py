from abc import ABC, abstractmethod


class EarlyStopper(ABC):
    @abstractmethod
    def reset(self):
        """Reset the early stopper"""

    @abstractmethod
    def __call__(self, *args, **kwargs) -> bool:
        """Return true if we should stop"""


class RelEarlyStopper(EarlyStopper):
    def __init__(self, *, wait=0, grace=0, change_threshold=0.01, direction="max"):
        """Early stopping based on relative changes in the objective value.

        Args:
            wait: Wait for this many iterations without improvement
            grace: Do not stop anything younger than this.
            change_threshold: Relative change in objective value to be considered an
                improvement
            direction: "min" or "max"
        """
        self._best_fom = None
        self._direction = direction
        self._i_const = 0
        self._i_report = 0
        self._change_threshold = change_threshold
        self._grace = grace
        self._wait = wait

    def reset(self):
        self._i_const = 0
        self._i_report = 0

    def __call__(self, fom: float) -> bool:
        self._i_report += 1
        if self._direction == "max":
            pass
        elif self._direction == "min":
            fom = -fom
        else:
            msg = f"Unknown direction {self._direction}"
            raise ValueError(msg)
        if self._best_fom is None:
            self._best_fom = fom
            return False
        if fom >= (1 + self._change_threshold) * self._best_fom:
            self._i_const = 0
            self._best_fom = fom
        else:
            self._i_const += 1
        if self._i_report > self._grace and self._i_const > self._wait:
            return True
        return False


class NoEarlyStopping(EarlyStopper):
    def __init__(self, *args, **kwargs):
        """Never stop early"""

    def reset(self):
        pass

    def __call__(self, fom: float) -> bool:
        return False


no_early_stopping = NoEarlyStopping()
