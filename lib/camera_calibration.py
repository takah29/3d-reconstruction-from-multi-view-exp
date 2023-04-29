from itertools import product
import numpy as np


def _get_observation_matrix(*data_list):
    length_list = [len(x) for x in data_list]
    if length_list.count(length_list[0]) != len(length_list):
        raise ValueError()

    W = np.hstack(data_list).T
    W -= W.mean(axis=1)[:,np.newaxis]

    return W


def _factorization(W):
    """観測行列Wから運動行列Mと形状行列Sを求める"""
    U, Sigma, Vt = np.linalg.svd(W)

    M = U[:, :3]
    S = np.diag(Sigma[:3]) @ Vt[:3]

    return M, S


def __factorization_by_orthographic_projection(W):
    """観測行列Wから運動行列Mと形状行列Sを求める

    平行投影カメラモデルを仮定する
    """
    U, Sigma, Vt = np.linalg.svd(W)

    U_ = U[:, :3]
    u_ = [U_[i : i + 2] for i in range(0, U_.shape[0], 2)]

    n_images = W.shape[0] // 2
    B_cal = np.zeros((3, 3, 3, 3))
    for n in range(n_images):
        for (i, j, k, l) in product(range(3), repeat=4):
            B_cal[i, j, k, l] += (
                u_[n][0, i] * u_[n][0, j] * u_[n][0, k] * u_[n][0, l]
                + u_[n][1, i] * u_[n][1, j] * u_[n][1, k] * u_[n][1, l]
                + 0.25
                * (u_[n][0, i] * u_[n][1, j] + u_[n][1, i] * u_[n][0, j])
                * (u_[n][0, k] * u_[n][1, l] + u_[n][1, k] * u_[n][0, l])
            )

    B1 = np.zeros((3, 3))
    B2 = np.zeros((3, 3))
    B3 = np.zeros((3, 3))
    B4 = np.zeros((3, 3))
    for (i, j) in product(range(3), repeat=2):
        B1[i, j] = B_cal[i, i, j, j]
        B2[i, j] = np.sqrt(2) * B_cal[i, i, (j + 1) % 3, (j + 2) % 3]
        B3[i, j] = np.sqrt(2) * B_cal[(i + 1) % 3, (i + 2) % 3, j, j]
        B4[i, j] = 2 * B_cal[(i + 1) % 3, (i + 2) % 3, (j + 1) % 3, (j + 2) % 3]
    B = np.block([[B1, B2], [B3, B4]])
    tau = np.linalg.inv(B) @ np.array([1, 1, 1, 0, 0, 0])

    T = np.array(
        [
            [tau[0], tau[5] / np.sqrt(2), tau[4] / np.sqrt(2)],
            [tau[5] / np.sqrt(2), tau[1], tau[3] / np.sqrt(2)],
            [tau[4] / np.sqrt(2), tau[3] / np.sqrt(2), tau[2]],
        ]
    )

    if np.linalg.det(T) < 0:
        T *= -1

    A = np.linalg.cholesky(T)
    M = U_ @ A
    S = np.linalg.inv(A) @ np.diag(Sigma[:3]) @ Vt[:3]

    return M, S


def affine_reconstruction(*data_list, method="orthographic"):
    W = _get_observation_matrix(*data_list)

    if method == "orthographic":
        M, S = __factorization_by_orthographic_projection(W)
    elif method == "simple":
        M, S = _factorization(W)
    else:
        raise ValueError()

    M_list = [M[i : i + 2] for i in range(0, M.shape[0], 2)]

    return M_list, S.T
