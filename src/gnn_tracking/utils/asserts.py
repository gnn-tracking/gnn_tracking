from torch import Tensor as T


def assert_feat_dim(feat_vec: T, dim: int) -> None:
    assert (
        feat_vec.shape[-1] == dim
    ), f"Expected feature dimension {dim}, got {feat_vec.shape[-1]}"
