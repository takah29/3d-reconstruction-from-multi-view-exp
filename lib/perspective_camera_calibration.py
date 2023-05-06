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

        # (n_points, n_images)
        res = []
        for i, ind in enumerate(max_eigvals_ind):
            # res.append(eigvecs[np.arange(n_points)[:, np.newaxis], :, max_eigvals_ind])
            res.append(eigvecs[i][:, ind])

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


def reconstruct_3d(x_list, f0):
    X = _create_data_matrix(x_list, f0)
    z = _compute_projective_depth(X, f0)

    # (n_points, n_images, 3) * (n_points, n_images, 1) -> (n_points, n_images, 3)
    W = X * z[..., np.newaxis]

    M, S = factorization_method(W.reshape(W.shape[0], -1).T)

    return M, S
