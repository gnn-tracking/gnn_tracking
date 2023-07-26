import numpy as np
from sklearn.cluster._dbscan_inner import dbscan_inner
from sklearn.neighbors import NearestNeighbors


class DBSCANFastRescan:
    def __init__(
        self, x: np.ndarray, max_eps: float = 1.0, *, n_jobs: int | None = None
    ):
        """Class to perform DBSCAN clustering with fast rescanning.

        Args:
            x: Data to cluster
            max_eps: Maximum epsilon to use during rescanning. Set to as low
                as possible to save time.
            n_jobs: The number of parallel jobs to run.
        """
        self.x = x
        self._max_eps = max_eps
        self._n_jobs = n_jobs
        self._distances = None
        self._edges = None
        self._reset_graph(max_eps)

    def _reset_graph(self, max_eps: float) -> None:
        """Set and store the radius_neighbors graph to use for clustering."""
        neighbors_model = NearestNeighbors(radius=max_eps, n_jobs=self._n_jobs)
        neighbors_model.fit(self.x)
        distances, indexes = neighbors_model.radius_neighbors(
            self.x, radius=max_eps, return_distance=True
        )

        lengths = [len(arr) for arr in indexes]
        source_indexes = np.repeat(np.arange(len(indexes)), lengths)
        target_indexes = np.concatenate(indexes)

        self._distances = np.concatenate(distances)
        self._edges = np.column_stack((source_indexes, target_indexes))

    def cluster(self, eps: float = 1.0, min_pts: int = 1):
        """Perform clustering on given data with DBSCAN

        Args:
            eps: Epsilon to use for clustering
            min_pts: Minimum number of points to form a cluster
        """
        if eps > self._max_eps:
            self._reset_graph(eps)

        filtered_edges = self._edges[self._distances <= eps]

        n_neighbors = np.bincount(filtered_edges[:, 0])
        core_samples = np.asarray(n_neighbors >= min_pts, dtype=np.uint8)

        idx = np.where(np.diff(filtered_edges[:, 0]) != 0)[0] + 1
        n = np.split(filtered_edges[:, 1], idx)
        neighborhoods = np.empty(len(n), dtype=object)

        for i in range(len(n)):
            neighborhoods[i] = n[i]

        labels = np.full(len(self.x), -1, dtype=np.intp)

        dbscan_inner(core_samples, neighborhoods, labels)

        return labels
