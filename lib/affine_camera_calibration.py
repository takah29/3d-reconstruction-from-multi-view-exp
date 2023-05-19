from itertools import product

import numpy as np
import numpy.typing as npt


def orthographic_self_calibration(
    data_list: list[npt.NDArray[np.floating]],
) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
    """平行投影カメラモデルによる自己校正を行う

    S.shape: (3, n_feature_points)
    R.shape: (image_num, 3, 3)
    """
    # 観測行列Wと画像中心tの取得
    W, t = _get_observation_matrix(data_list)

    # 因子分解
    U, Sigma, Vt = np.linalg.svd(W)

    U_ = U[:, :3]

    def _create_B_cal(U_):
        u_ = [U_[i : i + 2] for i in range(0, U_.shape[0], 2)]

        n_images = W.shape[0] // 2
        B_cal = np.zeros((3, 3, 3, 3))
        for n in range(n_images):
            for i, j, k, l in product(range(3), repeat=4):
                B_cal[i, j, k, l] += (
                    u_[n][0, i] * u_[n][0, j] * u_[n][0, k] * u_[n][0, l]
                    + u_[n][1, i] * u_[n][1, j] * u_[n][1, k] * u_[n][1, l]
                    + 0.25
                    * (u_[n][0, i] * u_[n][1, j] + u_[n][1, i] * u_[n][0, j])
                    * (u_[n][0, k] * u_[n][1, l] + u_[n][1, k] * u_[n][0, l])
                )

        return B_cal

    B_cal = _create_B_cal(U_)

    B = _get_B(B_cal)
    tau = np.linalg.solve(B, np.array([1, 1, 1, 0, 0, 0]))
    T = _get_T(tau)

    if np.linalg.det(T) < 0:
        T *= -1

    A = np.linalg.cholesky(T)
    M = U_ @ A
    S = np.linalg.inv(A) @ np.diag(Sigma[:3]) @ Vt[:3]

    # カメラの回転行列を計算
    R = _compute_rotation_mat(M, U_, T, t)

    return S.T, R


def symmetric_affine_self_calibration(
    data_list: list[npt.NDArray[np.floating]],
) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
    """対象アフィンカメラモデルによる自己校正を行う

    S.shape: (3, n_feature_points)
    R.shape: (image_num, 3, 3)
    """
    # 観測行列Wと画像中心tの取得
    W, t = _get_observation_matrix(data_list)

    # 因子分解
    U, Sigma, Vt = np.linalg.svd(W)

    U_ = U[:, :3]

    def _create_B_cal(U_):
        u_ = [U_[i : i + 2] for i in range(0, U_.shape[0], 2)]
        a = t.prod(axis=1)
        c = t[:, 0] ** 2 - t[:, 1] ** 2

        n_images = W.shape[0] // 2
        B_cal = np.zeros((3, 3, 3, 3))
        for n in range(n_images):
            for i, j, k, l in product(range(3), repeat=4):
                B_cal[i, j, k, l] += (
                    a[n] ** 2
                    * (
                        u_[n][0, i] * u_[n][0, j] * u_[n][0, k] * u_[n][0, l]
                        + u_[n][1, i] * u_[n][1, j] * u_[n][1, k] * u_[n][1, l]
                        - u_[n][0, i] * u_[n][0, j] * u_[n][1, k] * u_[n][1, l]
                        - u_[n][1, i] * u_[n][1, j] * u_[n][0, k] * u_[n][0, l]
                    )
                    + 0.25
                    * c[n] ** 2
                    * (
                        u_[n][0, i] * u_[n][1, j] * u_[n][0, k] * u_[n][1, l]
                        + u_[n][1, i] * u_[n][0, j] * u_[n][0, k] * u_[n][1, l]
                        + u_[n][0, i] * u_[n][1, j] * u_[n][1, k] * u_[n][0, l]
                        + u_[n][1, i] * u_[n][0, j] * u_[n][1, k] * u_[n][0, l]
                    )
                    - 0.5
                    * a[n]
                    * c[n]
                    * (
                        u_[n][0, i] * u_[n][0, j] * u_[n][0, k] * u_[n][1, l]
                        + u_[n][0, i] * u_[n][0, j] * u_[n][1, k] * u_[n][0, l]
                        + u_[n][0, i] * u_[n][1, j] * u_[n][0, k] * u_[n][0, l]
                        + u_[n][1, i] * u_[n][0, j] * u_[n][0, k] * u_[n][0, l]
                        - u_[n][0, i] * u_[n][1, j] * u_[n][1, k] * u_[n][1, l]
                        - u_[n][1, i] * u_[n][0, j] * u_[n][1, k] * u_[n][1, l]
                        - u_[n][1, i] * u_[n][1, j] * u_[n][0, k] * u_[n][1, l]
                        - u_[n][1, i] * u_[n][1, j] * u_[n][1, k] * u_[n][0, l]
                    )
                )

        return B_cal

    B_cal = _create_B_cal(U_)

    B = _get_B(B_cal)
    L, P = np.linalg.eig(B)
    tau = P[:, np.argmin(L)]
    T = _get_T(tau)

    if np.linalg.det(T) < 0:
        T *= -1

    A = np.linalg.cholesky(T)
    M = U_ @ A
    S = np.linalg.inv(A) @ np.diag(Sigma[:3]) @ Vt[:3]

    # カメラの回転行列を計算
    R = _compute_rotation_mat(M, U_, T, t)

    return S.T, R


def paraperspective_self_calibration(
    data_list: list[npt.NDArray[np.floating]], f: npt.NDArray[np.floating]
) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
    """疑似平行投影カメラモデルによる自己校正を行う

    S.shape: (3, n_feature_points)
    R.shape: (image_num, 3, 3)
    """
    if len(data_list) != len(f):
        raise ValueError()

    # 観測行列Wと画像中心tの取得
    W, t = _get_observation_matrix(data_list)

    # 因子分解
    U, Sigma, Vt = np.linalg.svd(W)

    U_ = U[:, :3]

    def _create_B_cal(U_, f):
        u_ = [U_[i : i + 2] for i in range(0, U_.shape[0], 2)]
        alpha = 1 / (1 + t[:, 0] ** 2 / f**2)
        beta = 1 / (1 + t[:, 1] ** 2 / f**2)
        gamma = t.prod(axis=1) / f**2

        n_images = W.shape[0] // 2
        B_cal = np.zeros((3, 3, 3, 3))
        for n in range(n_images):
            for i, j, k, l in product(range(3), repeat=4):
                B_cal[i, j, k, l] += (
                    (gamma[n] ** 2 + 1)
                    * alpha[n] ** 2
                    * (u_[n][0, i] * u_[n][0, j] * u_[n][0, k] * u_[n][0, l])
                    + (gamma[n] ** 2 + 1)
                    * beta[n] ** 2
                    * (u_[n][1, i] * u_[n][1, j] * u_[n][1, k] * u_[n][1, l])
                    + u_[n][0, i] * u_[n][1, j] * u_[n][0, k] * u_[n][1, l]
                    + u_[n][0, i] * u_[n][1, j] * u_[n][1, k] * u_[n][0, l]
                    + u_[n][1, i] * u_[n][0, j] * u_[n][0, k] * u_[n][1, l]
                    + u_[n][1, i] * u_[n][0, j] * u_[n][1, k] * u_[n][0, l]
                    - alpha[n]
                    * gamma[n]
                    * (
                        u_[n][0, i] * u_[n][0, j] * u_[n][0, k] * u_[n][1, l]
                        + u_[n][0, i] * u_[n][0, j] * u_[n][1, k] * u_[n][0, l]
                        + u_[n][0, i] * u_[n][1, j] * u_[n][0, k] * u_[n][0, l]
                        + u_[n][1, i] * u_[n][0, j] * u_[n][0, k] * u_[n][0, l]
                    )
                    - beta[n]
                    * gamma[n]
                    * (
                        u_[n][1, i] * u_[n][1, j] * u_[n][0, k] * u_[n][1, l]
                        + u_[n][1, i] * u_[n][1, j] * u_[n][1, k] * u_[n][0, l]
                        + u_[n][0, i] * u_[n][1, j] * u_[n][1, k] * u_[n][1, l]
                        + u_[n][1, i] * u_[n][0, j] * u_[n][1, k] * u_[n][1, l]
                    )
                    + (gamma[n] ** 2 - 1)
                    * alpha[n]
                    * beta[n]
                    * (
                        u_[n][0, i] * u_[n][0, j] * u_[n][1, k] * u_[n][1, l]
                        + u_[n][1, i] * u_[n][1, j] * u_[n][0, k] * u_[n][0, l]
                    )
                )

        return B_cal

    B_cal = _create_B_cal(U_, f)

    B = _get_B(B_cal)
    L, P = np.linalg.eig(B)
    tau = P[:, np.argmin(L)]
    T = _get_T(tau)

    if np.linalg.det(T) < 0:
        T *= -1

    A = np.linalg.cholesky(T)
    M = U_ @ A
    S = np.linalg.inv(A) @ np.diag(Sigma[:3]) @ Vt[:3]

    # カメラの回転行列を計算
    R = _compute_rotation_mat(M, U_, T, t)

    return S.T, R


def _get_observation_matrix(
    data_list: list[npt.NDArray[np.floating]],
) -> tuple[npt.NDArray, npt.NDArray]:
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


def _get_B(B_cal: npt.NDArray) -> npt.NDArray:
    """行列Bを取得する"""
    B1 = np.zeros((3, 3))
    B2 = np.zeros((3, 3))
    B3 = np.zeros((3, 3))
    B4 = np.zeros((3, 3))
    for i, j in product(range(3), repeat=2):
        B1[i, j] = B_cal[i, i, j, j]
        B2[i, j] = np.sqrt(2) * B_cal[i, i, (j + 1) % 3, (j + 2) % 3]
        B3[i, j] = np.sqrt(2) * B_cal[(i + 1) % 3, (i + 2) % 3, j, j]
        B4[i, j] = 2 * B_cal[(i + 1) % 3, (i + 2) % 3, (j + 1) % 3, (j + 2) % 3]
    B = np.block([[B1, B2], [B3, B4]])

    return B


def _get_T(tau: npt.NDArray) -> npt.NDArray:
    """計量行列Tを取得する"""
    T = np.array(
        [
            [tau[0], tau[5] / np.sqrt(2), tau[4] / np.sqrt(2)],
            [tau[5] / np.sqrt(2), tau[1], tau[3] / np.sqrt(2)],
            [tau[4] / np.sqrt(2), tau[3] / np.sqrt(2), tau[2]],
        ]
    )

    return T


def _get_zeta_beta_g(
    U_: npt.NDArray, T: npt.NDArray, t: npt.NDArray
) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
    image_num = t.shape[0]

    P = np.ones((image_num, 3, 2))
    P[:, :2, 1] = t**2
    P[:, 2, 0] = 0.0
    P[:, 2, 1] = t.prod(axis=1)

    U1 = U_[::2]
    U2 = U_[1::2]

    Q = np.zeros((image_num, 3))
    # (image_num, 1, 3) @ (1, 3, 3) @ (image_num, 3, 1) -> (image_num, 1, 1)
    Q[:, 0] = (U1[:, np.newaxis] @ T[np.newaxis] @ U1[..., np.newaxis]).ravel()
    Q[:, 1] = (U1[:, np.newaxis] @ T[np.newaxis] @ U2[..., np.newaxis]).ravel()
    Q[:, 2] = (U2[:, np.newaxis] @ T[np.newaxis] @ U2[..., np.newaxis]).ravel()

    # (image_num, 2, 3) @ (image_num, 3, 1) -> (image_num, 2, 1) -> (2, image_num)
    zeta2_inv, beta2 = (np.linalg.pinv(P) @ Q[..., np.newaxis]).squeeze(2).T

    # beta^2 < 0.0 のケース
    beta2[beta2 < 0.0] = 0.0

    # tx ~ 0.0 かつ ty ~ 0.0 のケース
    satisfied = (np.abs(t) < 1e-8).all(axis=1)
    beta2[satisfied] = 0.0
    zeta2_inv[satisfied] = ((Q[:, 0] + Q[:, 2]) / 2)[satisfied]
    zeta2_inv[zeta2_inv <= 0.0] = 1e8

    zeta = np.sqrt(1 / zeta2_inv)
    beta = np.sqrt(beta2)

    # (image_num,1) * (image_num, 2)
    g = zeta[:, np.newaxis] * t

    return zeta, beta, g


def _compute_rotation_mat(
    M: npt.NDArray, U_: npt.NDArray, T: npt.NDArray, t: npt.NDArray
) -> npt.NDArray:
    """カメラの回転行列を計算する"""

    zeta, beta, g = _get_zeta_beta_g(U_, T, t)

    # (image_num, 1) * (image_num, 3) - (image_num, 1) * ((image_num, 1, 2) @ (image_num, 2, 3))
    # -> (image_num, 3)
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
