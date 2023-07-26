import numpy as np
from sklearn.cluster import DBSCAN

from gnn_tracking.postprocessing.fastrescanner import DBSCANFastRescan


def test_fastrescan_equal_slowrescan():
    x = np.random.uniform(size=(100, 2))
    fr = DBSCANFastRescan(x, max_eps=0.15)
    for eps in [0.1, 0.05]:
        for min_pts in [1, 2]:
            labels = fr.cluster(eps=eps, min_pts=min_pts)
            labels2 = DBSCAN(eps=eps, min_samples=min_pts).fit_predict(x)
            assert (labels == labels2).all()
