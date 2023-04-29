from itertools import product
import numpy as np


def _get_observation_matrix(*data_list):
    """観測行列と各画像の重心ベクトルを計算する

    W.shape: (2 * image_num, n_feature_points)
    t.shape: (image_num, 2)
    """
    length_list = [len(x) for x in data_list]
    if length_list.count(length_list[0]) != len(length_list):
        raise ValueError()

    W = np.hstack(data_list).T
    t = W.mean(axis=1)[:, np.newaxis]
    W -= t

    return W, t.reshape(-1, 2)


def factorization_method(W):
    """観測行列Wから因子分解法によって、運動行列Mと形状行列Sを求める"""
    U, Sigma, Vt = np.linalg.svd(W)

    M = U[:, :3]
    S = np.diag(Sigma[:3]) @ Vt[:3]

    return M, S


def orthographic_self_calibration(*data_list):
    """平行投影カメラモデルによる自己校正を行う

    S.shape: (3, n_feature_points)
    R.shape: (image_num, 3, 3)
    """
    # 観測行列Wと画像中心tの取得
    W, t = _get_observation_matrix(*data_list)

    # 因子分解
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

    # カメラの回転行列を計算
    R = _compute_rotation_mat(M, U_, T, t, method="orthographic")

    return S.T, R


def _get_zeta_beta_g_for_orthographic(M, t):
    zeta = np.ones(M.shape[0] // 2)
    beta = np.zeros(M.shape[0] // 2)
    g = t

    return zeta, beta, t


def _compute_rotation_mat(M, U, T, t, method="orthographic"):
    """U, T, tから回転行列を計算する"""
    # (image_num, )
    if method == "orthographic":
        zeta, beta, g = _get_zeta_beta_g_for_orthographic(M, t)
    else:
        raise ValueError()

    # (image_num, 1) * (image_num, 3) - (image_num, 1) * ((image_num, 1, 2) @ (image_num, 2, 3)) -> (image_num, 3)
    r3_denom = zeta[..., np.newaxis] * np.cross(M[::2], M[1::2]) - beta[..., np.newaxis] * (
        g[:, np.newaxis] @ M.reshape(-1, 2, 3)
    ).squeeze(1)
    # (image_num, 1) * (image_num, 1, 2) @ (image_num, 2, 1) -> (image_num, 1)
    r3_num = 1 + beta[..., np.newaxis] ** 2 * (g.reshape(-1, 1, 2) @ g.reshape(-1, 2, 1))[0]
    # (image_num, 3) / (image_num, 1) -> (image_num, 3)
    r3 = r3_denom / r3_num

    # (image_num, 1) * (image_num, 3) + (image_num, 1) * (image_num, 3) -> (image_num, 3)
    r1 = zeta[:, np.newaxis] * M[::2] + (beta * g[:, 0])[:, np.newaxis] * r3

    # (image_num, 1) * (image_num, 3) + (image_num, 1) * (image_num, 3) -> (image_num, 3)
    r2 = zeta[:, np.newaxis] * M[1::2] + (beta * g[:, 1])[:, np.newaxis] * r3

    R = np.hstack((r1, r2, r3)).reshape(-1, 3, 3).transpose(0, 2, 1)

    # 厳密な回転行列に補正する
    U, _, Vt = np.linalg.svd(R)
    R = U @ Vt

    return R