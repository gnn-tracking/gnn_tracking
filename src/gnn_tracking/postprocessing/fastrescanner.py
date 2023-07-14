import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster._dbscan_inner import dbscan_inner
from sklearn.cluster import DBSCAN

class DBSCANFastRescan:

    def __init__(self, X, max_eps=1, n_jobs=None):
        """Class to perform DBSCAN clustering with fast rescanning.

        Args:
            X: Data to cluster
            max_eps: Maximum epsilon to use during rescanning. Set to as low
                as possible to save time.
            n_jobs : int, default=None
                The number of parallel jobs to run.
        """
        self.X = X
        self.max_eps = max_eps
        self.n_jobs=n_jobs
        self._reset_graph(max_eps)

    def _reset_graph(self, max_eps):
        """
        Set and store the radius_neighbors graph to use for clustering.
        """
        neighbors_model = NearestNeighbors(radius=max_eps)
        neighbors_model.fit(self.X)
        distances, indexes = neighbors_model.radius_neighbors(self.X, radius=max_eps, return_distance=True, n_jobs=self.n_jobs)

        lengths = [len(arr) for arr in indexes]
        source_indexes = np.repeat(np.arange(len(indexes)), lengths)
        target_indexes = np.concatenate(indexes)

        self.distances = np.concatenate(distances)
        self.edges = np.column_stack((source_indexes, target_indexes))
    
    def cluster(self, eps = 1, min_pts= 1):
        """Perform clustering on given data with DBSCAN

        Args:
            eps: Epsilon to use for clustering
            min_pts: Minimum number of points to form a cluster
        """
        if(eps > self.max_eps):
            self._reset_graph(eps)

        filtered_edges = self.edges[self.distances <= eps]

        n_neighbors = np.bincount(filtered_edges[:,0])
        core_samples = np.asarray(n_neighbors >= min_pts, dtype=np.uint8)

        idx = np.where(np.diff(filtered_edges[:,0]) != 0)[0] + 1
        n = np.split(filtered_edges[:,1], idx)
        neighborhoods = np.empty(len(n), dtype=object)

        for i in range(len(n)):
            neighborhoods[i] = n[i]

        labels = np.full(len(self.X), -1, dtype=np.intp)

        dbscan_inner(core_samples, neighborhoods, labels)

        return labels
