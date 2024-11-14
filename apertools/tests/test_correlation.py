import numpy as np
from apertools import correlation


def test_cov_matrix_tropo():
    ifglist = [(1, 2), (1, 3), (2, 3)]
    sar_variances = [1, 1, 1]
    out = correlation.cov_matrix_tropo(ifglist, sar_variances)
    assert out.shape == (3, 3)
    assert np.allclose(
        out,
        np.array([
            [2, 1, -1],
            [1, 2, 1],
            [-1, 1, 2],
        ]),
    )

    sar_variances = [1, 4, 9]
    out = correlation.cov_matrix_tropo(ifglist, sar_variances)
    assert out.shape == (3, 3)
    assert np.allclose(
        out,
        np.array([
            [5, 1, -4],
            [1, 10, 9],
            [-4, 9, 13],
        ]),
    )
