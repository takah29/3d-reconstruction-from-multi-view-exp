import numpy as np
from numpy.typing import NDArray


def factorization_method(
    W: NDArray[np.floating], n_rank: int = 4
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """観測行列Wから因子分解法によって、運動行列Mと形状行列Sを求める"""

    U, Sigma, Vt = np.linalg.svd(W)

    M = U[:, :n_rank]
    S = np.diag(Sigma[:n_rank]) @ Vt[:n_rank]

    return M, S
