from itertools import product
from typing import Tuple

import numpy as np
import numpy.typing as npt


def factorization_method(
    W: npt.NDArray[np.floating], n_rank=4
) -> Tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
    """観測行列Wから因子分解法によって、運動行列Mと形状行列Sを求める"""
    U, Sigma, Vt = np.linalg.svd(W)

    M = U[:, :n_rank]
    S = np.diag(Sigma[:n_rank]) @ Vt[:n_rank]

    return M, S


def _get_initial_inner_camera_params(n_images, f0):
    """初期内部カメラパラメータ行列Kを作成する"""
    return np.tile(np.eye(3) * f0, (n_images, 1, 1))


def _create_data_matrix(x_list, f0):
    # (n_images, n_points, 3)
    X = np.asarray([np.hstack((x / f0, np.ones((x.shape[0], 1)))) for x in x_list])
    # (n_points, n_images, 3)
    X = X.transpose(1, 0, 2)

    return X


def _compute_reprojection_error(X, M, S, f0):
    """再投影誤差を計算する"""
    # (3 * n_images, 4) @ (4, n_points) -> (3 * n_images, n_points) -> (n_images, 3, n_points)
    # -> (n_points, n_images, 3)
    PX = (M @ S).reshape(-1, 3, S.shape[1]).transpose(2, 0, 1)

    # 第3成分を1にする正規化 (a, b, c) -> (a/c, b/c, 1) を行う
    PX = np.apply_along_axis(lambda x: x / x[2], 2, PX)

    X_minus_PX = X - PX

    # (n_points, n_images, 1, 3) @ (n_points, n_images, 3, 1) -> (n_points, n_images, 1, 1)
    # -> (n_points, n_images) -> ()
    E = f0 * np.sqrt((X_minus_PX[:, :, np.newaxis] @ X_minus_PX[..., np.newaxis]).squeeze().mean())

    return E


def _compute_projective_depth(
    X, f0: float, tolerance: float = 2.0, max_iter: int = 100
) -> npt.NDArray:
    """データXから射影的奥行きzを求める

    Args:
        X (npt.NDArray): アフィンカメラを仮定した観測行列をもとにしたデータ, X.shape = (n_points, n_images, 3)
        tolerance (float, optional): 許容再投影誤差

    Returns:
        npt.NDArray: 射影的奥行き, z.shape = (n_images, n_feature_points)
    """
    n_points = X.shape[0]
    n_images = X.shape[1]

    z = np.ones((n_points, n_images))

    count = 0
    while True:
        # (n_points, n_images, 3) * (n_points, n_images, 1) -> (n_points, n_images, 3)
        W = X * z[..., np.newaxis]

        # Wの各列を単位ベクトルにする
        # (n_points, n_images, 3) / (n_points, 1, 1) -> (n_points, n_images, 3)
        W = W / np.linalg.norm(W, axis=(1, 2))[:, np.newaxis, np.newaxis]

        U, Sigma, Vt = np.linalg.svd(W.reshape(n_points, -1).T)

        # (3 * n_images, min(2 * n_images, n_points)) -> (3 * n_images, 4) -> (n_images, 3, 4)
        # -> (4, n_images, 3)
        U_ = U[:, :4].reshape(n_images, 3, 4).transpose(2, 0, 1)

        # (n_points, 1, n_images, 1, 3) @ (1, 4, n_images, 3, 1) -> (n_points, 4, n_images, 1, 1)
        # -> (n_points, 4, n_images)
        x_dot_u = (X[:, np.newaxis, :, np.newaxis, :] @ U_[np.newaxis, ..., np.newaxis]).squeeze()

        # (n_points, 4, n_images, 1) @ (n_points, 4, 1, n_images)
        # -> (n_points, 4, n_images, n_images) -> (n_points, n_images, n_images)
        denom = (x_dot_u[..., np.newaxis] @ x_dot_u[:, :, np.newaxis]).sum(axis=1)

        # (n_points, n_images)
        X_norm = np.linalg.norm(X, axis=2)

        # (n_points, n_images, 1) @ (n_points, 1, n_images) -> (n_points, n_images, n_images)
        num = X_norm[..., np.newaxis] @ X_norm[:, np.newaxis]

        # (n_points, n_images, n_images) / (n_points, n_images, n_images)
        # -> (n_points, n_images, n_images)
        A = denom / num

        # (n_points, n_images), (n_points, n_images, n_images)
        eigvals, eigvecs = np.linalg.eig(A)

        # (n_points, n_images) -> (n_points, )
        max_eigvals_ind = np.argmax(eigvals, axis=1)

        res = []
        for i, ind in enumerate(max_eigvals_ind):
            res.append(eigvecs[i][:, ind])

        # (n_points, n_images)
        xi = np.real(np.vstack(res))

        # sum(xi[i]) < 0 の場合はxi[i]の符号を反転する
        xi[xi.sum(axis=1) < 0] *= -1

        # (n_points, n_images) / (n_points, n_images) -> (n_points, n_images)
        z[...] = xi / X_norm

        M = U_.transpose(1, 2, 0).reshape(-1, 4)
        S = np.diag(Sigma[:4]) @ Vt[:4]

        E = _compute_reprojection_error(X, M, S, f0)

        count += 1

        if E < tolerance or count >= max_iter:
            break

    if count >= max_iter:
        print("Did not converge because the maximum number of iterations was reached.")

    return z


def _compute_projective_depth_dual_method(
    X, f0: float, tolerance: float = 2.0, max_iter: int = 100
) -> npt.NDArray:
    """データXから双対法で射影的奥行きzを求める

    Args:
        X (npt.NDArray): アフィンカメラを仮定した観測行列をもとにしたデータ, X.shape = (n_points, n_images, 3)
        tolerance (float, optional): 許容再投影誤差

    Returns:
        npt.NDArray: 射影的奥行き, z.shape = (n_images, n_feature_points)
    """
    n_points = X.shape[0]
    n_images = X.shape[1]

    z = np.ones((n_points, n_images))

    count = 0
    while True:
        # (n_points, n_images, 3) * (n_points, n_images, 1) -> (n_points, n_images, 3)
        W = X * z[..., np.newaxis]

        # Wの各列を単位ベクトルにする
        # (n_images, 3, n_points)
        X = X.transpose(1, 2, 0)
        # (n_images, 3, n_points) / (n_images, 1, 1) -> (n_images, 3, n_points)
        # -> (n_points, n_images, 3)
        W = (X / (np.linalg.norm(X, axis=2) ** 2).sum(axis=1)[:, np.newaxis, np.newaxis]).transpose(
            2, 0, 1
        )

        U, Sigma, Vt = np.linalg.svd(W.reshape(n_points, -1).T)

        # (min(3 * n_images, n_points), n_points) -> (4, n_points) -> (n_points, 4)
        V_ = Vt[:4].T

        # (n_points, 4) @ (4, n_points) -> (n_points, n_points)
        V_gram_mat = V_ @ V_.T

        # (n_images, n_points, 3) @ (n_images, 3, n_points) -> (n_images, n_points, n_points)
        X_gram_mat = X.transpose(0, 2, 1) @ X

        # (n_points, n_points) @ (n_images, n_points, n_points) -> (n_images, n_points, n_points)
        denom = V_gram_mat * X_gram_mat

        # (n_images, 3, n_points) -> (n_images, n_points)
        X_norm = np.linalg.norm(X, axis=1)

        # (n_images, n_points, 1) @ (n_images, 1, n_points) -> (n_images, n_points, n_points)
        num = X_norm[..., np.newaxis] @ X_norm[:, np.newaxis]

        # (n_images, n_points, n_points) / (n_images, n_points, n_points)
        # -> (n_images, n_points, n_points)
        B = denom / num

        # (n_images, n_points), (n_images, n_points, n_points)
        eigvals, eigvecs = np.linalg.eig(B)

        # (n_images, n_points) -> (n_images, )
        max_eigvals_ind = np.argmax(eigvals, axis=1)

        res = []
        for i, ind in enumerate(max_eigvals_ind):
            res.append(eigvecs[i][:, ind])

        # (n_points, n_images)
        xi = np.real(np.vstack(res)).T

        # sum(xi[i]) < 0 の場合はxi[i]の符号を反転する
        xi[xi.sum(axis=1) < 0] *= -1

        # (n_points, n_images) / (n_points, n_images) -> (n_points, n_images)
        z[...] = xi / X_norm.T

        M = U[:, :4]
        S = np.diag(Sigma[:4]) @ V_.T
        E = _compute_reprojection_error(X.transpose(2, 0, 1), M, S, f0)

        count += 1

        if E < tolerance or count >= max_iter:
            break

    if count >= max_iter:
        print("Did not converge because the maximum number of iterations was reached.")

    return z


def _calc_omega(Q):
    def _create_A_cal(Q):
        n_images = Q.shape[0]
        A_cal = np.zeros((4, 4, 4, 4))
        for n in range(n_images):
            for (i, j, k, l) in product(range(4), repeat=4):
                A_cal[i, j, k, l] += (
                    Q[n, 0, i] * Q[n, 0, j] * Q[n, 0, k] * Q[n, 0, l]
                    - Q[n, 0, i] * Q[n, 0, j] * Q[n, 1, k] * Q[n, 1, l]
                    - Q[n, 1, i] * Q[n, 1, j] * Q[n, 0, k] * Q[n, 0, l]
                    + Q[n, 1, i] * Q[n, 1, j] * Q[n, 1, k] * Q[n, 1, l]
                    + 0.25
                    * (
                        Q[n, 0, i] * Q[n, 1, j] * Q[n, 0, k] * Q[n, 1, l]
                        + Q[n, 1, i] * Q[n, 0, j] * Q[n, 0, k] * Q[n, 1, l]
                        + Q[n, 0, i] * Q[n, 1, j] * Q[n, 1, k] * Q[n, 0, l]
                        + Q[n, 1, i] * Q[n, 0, j] * Q[n, 1, k] * Q[n, 0, l]
                    )
                    + 0.25
                    * (
                        Q[n, 1, i] * Q[n, 2, j] * Q[n, 1, k] * Q[n, 2, l]
                        + Q[n, 2, i] * Q[n, 1, j] * Q[n, 1, k] * Q[n, 2, l]
                        + Q[n, 1, i] * Q[n, 2, j] * Q[n, 2, k] * Q[n, 1, l]
                        + Q[n, 2, i] * Q[n, 1, j] * Q[n, 2, k] * Q[n, 1, l]
                    )
                    + 0.25
                    * (
                        Q[n, 2, i] * Q[n, 0, j] * Q[n, 2, k] * Q[n, 0, l]
                        + Q[n, 0, i] * Q[n, 2, j] * Q[n, 2, k] * Q[n, 0, l]
                        + Q[n, 2, i] * Q[n, 0, j] * Q[n, 0, k] * Q[n, 2, l]
                        + Q[n, 0, i] * Q[n, 2, j] * Q[n, 0, k] * Q[n, 2, l]
                    )
                )

        return A_cal

    def _get_A(A_cal):
        A1 = np.zeros((4, 4))
        for i, j in product(range(4), repeat=2):
            A1[i, j] = A_cal[i, i, j, j]

        ind_list = [(i1, i2) for i1 in range(4) for i2 in range(i1 + 1, 4)]
        A2 = np.zeros((4, 6))
        A3 = np.zeros((6, 4))
        for i in range(4):
            for j, (j1, j2) in enumerate(ind_list):
                A2[i, j] = np.sqrt(2) * A_cal[i, i, j1, j2]
                A3[j, i] = np.sqrt(2) * A_cal[j1, j2, i, i]

        A4 = np.zeros((6, 6))
        for i, (i1, i2) in enumerate(ind_list):
            for j, (j1, j2) in enumerate(ind_list):
                A4[i, j] = 2 * A_cal[i1, i2, j1, j2]

        A = np.block([[A1, A2], [A3, A4]])

        return A

    def _get_Omega(omega):
        sqrt2 = np.sqrt(2)
        Omega = np.array(
            [
                [omega[0], omega[4] / sqrt2, omega[5] / sqrt2, omega[6] / sqrt2],
                [omega[4] / sqrt2, omega[1], omega[7] / sqrt2, omega[8] / sqrt2],
                [omega[5] / sqrt2, omega[7] / sqrt2, omega[2], omega[9] / sqrt2],
                [omega[6] / sqrt2, omega[8] / sqrt2, omega[9] / sqrt2, omega[3]],
            ]
        )

        return Omega

    A_cal = _create_A_cal(Q)
    A = _get_A(A_cal)
    eigvals, eigvecs = np.linalg.eig(A)
    omega = eigvecs[:, np.argmin(eigvals)]
    Omega = _get_Omega(omega)

    eigvals, eigvecs = np.linalg.eig(Omega)

    max_eigvals_ind = np.argsort(eigvals)[::-1]
    sigma = eigvals[max_eigvals_ind]
    res = []
    for i in max_eigvals_ind:
        res.append(eigvecs[:, i])

    w = np.vstack(res)

    if sigma[2] > 0:
        # (4, 3) @ (3, 4) -> (4, 4)
        Omega = (sigma[:3, np.newaxis] * w[:3]).T @ w[:3]
    elif sigma[1] < 0:
        # (4, 3) @ (3, 4) -> (4, 4)
        Omega = -((sigma[2:, np.newaxis] * w[2:]).T @ w[2:])
    else:
        raise ValueError()

    return Omega, sigma, w


def _update_K(K, Omega, Q):
    """内部カメラパラメータKを更新する"""
    # (n_images, 3, 4) @ (4, 4) @ (n_images, 4, 3) -> (n_images, 3, 3)
    C = Q @ Omega @ Q.transpose(0, 2, 1)

    # (n_images, )
    F = (
        (C[:, 0, 0] + C[:, 1, 1]) / C[:, 2, 2]
        - (C[:, 0, 2] / C[:, 2, 2]) ** 2
        - (C[:, 1, 2] / C[:, 2, 2]) ** 2
    )

    J = np.full(F.shape, np.inf)
    is_updatable = (C[:, 2, 2] > 0) & (F > 0)
    if is_updatable.any():
        delta_u0 = C[:, 0, 2] / C[:, 2, 2]
        delta_v0 = C[:, 1, 2] / C[:, 2, 2]
        delta_f = np.sqrt(
            0.5 * ((C[:, 0, 0] + C[:, 1, 1]) / C[:, 2, 2] - delta_u0**2 - delta_v0**2)
        )

        delta_K = np.zeros((3, 3, delta_f.shape[0]))
        delta_K[(0, 1), (0, 1)] = delta_f
        delta_K[0, 2] = delta_u0
        delta_K[1, 2] = delta_v0
        delta_K[2, 2] = 1
        delta_K = delta_K.transpose(2, 0, 1)

        # (n_images, 3, 3) @ (n_images, 3, 3) -> (n_images, 3, 3)
        K = K @ delta_K

        # (n_images, ) * (n_images, 3, 3)
        K = np.sqrt(C[:, 2, 2])[:, np.newaxis, np.newaxis] * K

        J[is_updatable] = (
            (C[:, 0, 0] / C[:, 2, 2] - 1) ** 2
            + (C[:, 1, 1] / C[:, 2, 2] - 1) ** 2
            + 2 * (C[:, 0, 1] ** 2 + C[:, 1, 2] ** 2 + C[:, 2, 0] ** 2) / (C[:, 2, 2] ** 2)
        )[is_updatable]

    return K, J


def _euclidean_upgrading(P: npt.NDArray, f0: float):
    n_images = P.shape[0]
    J_med_ = np.inf
    K = _get_initial_inner_camera_params(n_images, f0)

    while True:
        # (n_images, 3, 3) @ (n_images, 3, 4) -> (n_images, 3, 4)
        Q = np.linalg.inv(K) @ P

        Omega, omega_eigval, omega_eigvec = _calc_omega(Q)

        if omega_eigval[2] > 0:
            coef = np.hstack((np.sqrt(omega_eigval[:3]), [1.0]))
            H = (coef[:, np.newaxis] * omega_eigvec).T
        elif omega_eigval[1] < 0:
            coef = np.hstack(([1.0], np.sqrt(-omega_eigval[1:])))
            H = (coef[:, np.newaxis] * omega_eigvec)[::-1].T
        else:
            raise ValueError()

        K, J = _update_K(K, Omega, Q)
        J_med = np.median(J)

        if J_med < 1e-8 or J_med >= J_med_:
            break

        J_med_ = J_med

    return H, K


def perspective_self_calibration(x_list, f0):
    X = _create_data_matrix(x_list, f0)
    z = _compute_projective_depth(X, f0)

    # (n_points, n_images, 3) * (n_points, n_images, 1) -> (n_points, n_images, 3)
    W = X * z[..., np.newaxis]

    M, S = factorization_method(W.reshape(W.shape[0], -1).T)

    P = M.reshape(-1, 3, 4)
    H, K = _euclidean_upgrading(P, f0)

    X_ = (np.linalg.inv(H) @ S).T
    X_ = X_[:, :3] / X_[:, -1:]

    return X_
