# Loosely following the implementation from
# https://debuggercafe.com/using-learning-rate-scheduler-and-early-stopping-with-pytorch/
class StopEarly:
    def __init__(self, patience=5, min_delta=0):
        """

        Args:
            patience: Number of epochs of no improvement after which training will be
                stopped. If set to 0, early stopping is disabled.
            min_delta: Minimum change in loss that we would consider an improvement.
        """
        assert patience >= 0
        self.patience = patience
        assert min_delta >= 0
        self.min_delta = min_delta
        self._best_loss = None
        self._times_no_improvement = 0

    def __call__(self, loss: float) -> bool:
        """Returns True if training should be stopped."""
        if self.patience == 0:
            return False
        if self._best_loss is None:
            self._best_loss = loss
            return False
        elif self._best_loss - loss > self.min_delta:
            self._best_loss = loss
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


no_early_stopping = StopEarly(patience=0)
