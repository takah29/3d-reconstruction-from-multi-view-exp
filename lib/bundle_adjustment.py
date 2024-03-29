from typing import Any

import numpy as np
import numpy.typing as npt
from scipy.linalg import block_diag

from .utils import get_rotation_matrix


class BundleAdjuster:
    def __init__(
        self,
        x: npt.NDArray,
        init_X: npt.NDArray,
        init_K: npt.NDArray,
        init_R: npt.NDArray,
        init_t: npt.NDArray,
        f0: float = 1.0,
        visibility_index: npt.NDArray | None = None,
        axis: str = "x-right_z-forward",
    ):
        # 入力時の座標へ戻すためにカメラパラメータを保存しておく
        if axis == "x-right_z-forward":
            c0c1_len = np.abs(init_R[0, :, 0] @ (init_t[1] - init_t[0]))
        elif axis == "x-up_z-forward":
            c0c1_len = np.abs(init_R[0, :, 1] @ (init_t[1] - init_t[0]))
        else:
            raise ValueError()
        self._init_camera0_params = {
            "R": init_R[0],
            "t": init_t[0],
            "c0c1_len": c0c1_len,
        }

        # 最適化対称の変数を格納
        # (n_points, n_images, 2)
        self._x = x

        # (n_points, 3), (n_images, 3, 3), (n_images, 3)
        self._X, self._R, self._t = BundleAdjuster._transform_to_normalize_coodinates(
            init_X, init_R, init_t, axis=axis
        )

        # (n_images, )
        self._f = init_K[:, 0, 0]

        # (n_images, 2)
        self._u = init_K[:, :2, 2]

        self._f0 = f0

        self._n_points = init_X.shape[0]
        self._n_images = init_R.shape[0]

        # 可視性指標
        self._visibility_index = (
            visibility_index
            if visibility_index is not None
            else np.ones(self._x.shape[:2], dtype=np.bool_)
        )

        if axis == "x-right_z-forward":
            # カメラパラメータはR1=I, t1=0, t2_1=1を仮定しているので、該当の未知数を削除するためのインデックスを用意する
            # カメラパラメータ: (f1, u01, v01, t1_1, t1_2, t1_3, omega1_1, omega1_2, omega1_3, f2, u02, ...)
            self._remove_ind = np.array([3, 4, 5, 6, 7, 8, 12])

            # 削除したカメラパラメータの要素を挿入するためのインデックス
            self._insert_ind = np.array([3, 3, 3, 3, 3, 3, 6])
        elif axis == "x-up_z-forward":
            # カメラパラメータはR1=I, t1=0, t2_2=1を仮定しているので、該当の未知数を削除するためのインデックスを用意する
            self._remove_ind = np.array([3, 4, 5, 6, 7, 8, 13])
            self._insert_ind = np.array([3, 3, 3, 3, 3, 3, 7])

        # 最適化時にイテレーションごとの3次元点とカメラパラメータのログを保存する変数
        self._log: list[dict[str, npt.NDArray | float]] = []

    def optimize(
        self,
        scale_factor: float = 10.0,
        delta_tol: float = 1e-8,
        max_iter: int = 100,
        is_debug: bool = False,
    ) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray]:
        """再投影誤差を最小化するX, K, R, tを求める"""
        K = self._get_K(self._f, self._u)
        P, p, q, r = self._calc_pqr(self._X, K, self._R, self._t)
        E = self._calc_reprojection_error(p, q, r)

        if is_debug:
            self._log.clear()
            self._log.append(
                {
                    "points": self._X.copy(),
                    "basis": self._R.copy(),
                    "pos": self._t.copy(),
                    "reprojection_error": E,
                }
            )

        c = 0.0001
        count = 0
        while True:
            K = self._get_K(self._f, self._u)
            P, p, q, r = self._calc_pqr(self._X, K, self._R, self._t)

            dpdX, dqdX, drdX = self._calc_X_diff_pqr(P)
            dp_domega, dq_domega, dr_domega = self._calc_camera_params_diff_pqr(p, q, r)

            # 1階微分
            d_P = self._calc_d_P(p, q, r, dpdX, dqdX, drdX)
            d_F = self._calc_d_F(p, q, r, dp_domega, dq_domega, dr_domega)

            # 2階微分
            matE = self._calc_matE(p, q, r, dpdX, dqdX, drdX)
            matF = self._calc_matF(p, q, r, dpdX, dqdX, drdX, dp_domega, dq_domega, dr_domega)
            matG = self._calc_matG(p, q, r, dp_domega, dq_domega, dr_domega)

            while True:
                # 2階微分行列matE, matGの対角成分を1+c倍する
                matEc = matE.copy()
                tmp_ind = np.arange(3)
                matEc[:, tmp_ind, tmp_ind] *= 1 + c
                matGc = matG.copy()
                tmp_ind = np.arange(9 * self._n_images - 7)
                matGc[tmp_ind, tmp_ind] *= 1 + c

                # (n_points, 3, 3)
                matEinv = np.linalg.inv(matEc)

                # (n_points, 9 * n_images - 7, 3) @ (n_points, 3, 3)
                # -> (n_points, 9 * n_images - 7, 3)
                FtEinv = matF.transpose(0, 2, 1) @ matEinv

                # (9 * n_images - 7, 9 * n_images - 7)
                A = matGc - (FtEinv @ matF).sum(axis=0)

                # (n_points, 3, 1)
                delta_X_E = d_P.reshape(self._n_points, 3)[..., np.newaxis]

                # (n_points, 9 * n_images - 7, 3) @ (n_points, 3, 1)
                # -> (n_points, 9 * n_images - 7, 1)　-> (n_points, 9 * n_images - 7)
                # -> (9 * n_images - 7, )
                b = (FtEinv @ delta_X_E).squeeze().sum(axis=0) - d_F

                # (9 * n_images - 7, )
                delta_xi_F = np.linalg.solve(A, b)

                # (n_points, 3, 3) @
                # ((n_points, 3, 9 * n_images - 7) @ (9 * n_images - 7, 1) + (n_points, 3, 1))
                # -> (n_points, 3, 1)
                # -> (n_points, 3)
                delta_X = -(matEinv @ (matF @ delta_xi_F[:, np.newaxis] + delta_X_E)).squeeze()

                # パラメータの更新
                tmp_X = self._update_3d_points(delta_X)
                tmp_f, tmp_u, tmp_t, tmp_R = self._update_camera_params(delta_xi_F)

                # 更新したパラメータで再投影誤差を計算する
                tmp_K = self._get_K(tmp_f, tmp_u)
                _, tmp_p, tmp_q, tmp_r = self._calc_pqr(tmp_X, tmp_K, tmp_R, tmp_t)

                E_ = self._calc_reprojection_error(tmp_p, tmp_q, tmp_r)

                if E_ > E:
                    c *= scale_factor
                else:
                    break

            self._X = tmp_X
            self._f = tmp_f
            self._u = tmp_u
            self._t = tmp_t
            self._R = tmp_R

            if is_debug:
                self._log.append(
                    {
                        "points": self._X.copy(),
                        "basis": self._R.copy(),
                        "pos": self._t.copy(),
                        "reprojection_error": E_,
                    }
                )

            count += 1
            reprojection_error_delta = np.abs(E_ - E)

            print(f"Iteration {count}: reprojection_error_delta = {reprojection_error_delta}")

            # 再投影誤差が変化しないまたは、max_iterに達したら終了
            if reprojection_error_delta <= delta_tol or count >= max_iter:
                break
            else:
                E = E_
                c /= scale_factor

        # 最適化前座標系へ戻す
        self._X, self._R, self._t = BundleAdjuster._inverse_transform_to_global_coordinates(
            self._init_camera0_params, self._X, self._R, self._t
        )

        return self._X, self._get_K(self._f, self._u), self._R, self._t

    def get_log(self) -> list[dict[str, npt.NDArray | float]]:
        """イテレーションごとに記録した3次元点とカメラパラメータを取得する"""
        return self._log

    @staticmethod
    def _transform_to_normalize_coodinates(
        X: npt.NDArray,
        R: npt.NDArray,
        t: npt.NDArray,
        axis: str = "x-right_z-forward",
    ) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
        """第1カメラを基準にシーンを正規化する

        Args:
            X (npt.NDArray): 3次元点
            R (npt.NDArray): カメラの回転行列
            t (npt.NDArray): カメラの並進
        """

        X_ = X - t[0]
        t_ = t - t[0]

        # (3, ) @ (3, 3) @ (3, 1) -> (3, )
        if axis == "x-right_z-forward":
            j = np.array([np.sign(t_[1, 0]), 0, 0])
        elif axis == "x-up_z-forward":
            j = np.array([0, np.sign(t_[1, 1]), 0])
        else:
            raise ValueError()

        s = j @ R[0].T @ t_[1][:, np.newaxis]

        X_ = ((X_) @ R[0]) / s
        R_ = R[0].T @ R
        t_ = ((t_) @ R[0]) / s

        return X_, R_, t_

    @staticmethod
    def _inverse_transform_to_global_coordinates(
        camera0_param: dict[str, Any],
        X: npt.NDArray,
        R: npt.NDArray,
        t: npt.NDArray,
    ):
        """グローバル座標へ戻す"""
        R0 = camera0_param["R"]
        t0 = camera0_param["t"]
        scale = camera0_param["c0c1_len"]

        X_ = (scale * X) @ R0.T + t0
        t_ = (scale * t) @ R0.T + t0
        R_ = R0 @ R

        return X_, R_, t_

    def _update_3d_points(self, delta_X: npt.NDArray) -> npt.NDArray:
        return self._X + delta_X

    def _update_camera_params(
        self, delta_xi_F: npt.NDArray
    ) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray]:
        # (9 * n_images, )
        delta_xi_F = np.insert(delta_xi_F, self._insert_ind, np.zeros(self._insert_ind.shape))

        # (n_images, 9)
        delta_xi_F = delta_xi_F.reshape(self._n_images, 9)

        # (n_images, 1), (n_images, 2), (n_images, 3), (n_images, 3)
        delta_f, delta_u, delta_t, delta_omega = np.hsplit(delta_xi_F, [1, 3, 6])

        # (n_images, )
        delta_f = delta_f.squeeze()

        # (n_images, 3, 3)
        delta_R = np.stack([get_rotation_matrix(do) for do in delta_omega])

        return self._f + delta_f, self._u + delta_u, self._t + delta_t, delta_R @ self._R

    def _get_K(self, f: npt.NDArray, u: npt.NDArray) -> npt.NDArray:
        K = np.zeros((self._n_images, 3, 3))
        K[:, (0, 1), (0, 1)] = f[:, np.newaxis]
        K[:, :2, 2] = u
        K[:, 2, 2] = self._f0

        return K

    def _calc_pqr(
        self, X: npt.NDArray, K: npt.NDArray, R: npt.NDArray, t: npt.NDArray
    ) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray]:
        """カメラ行列Pとp, q, rの3次元位置Xに関する微分を求める"""
        # (n_points, 4)
        X_ext = np.hstack((X, np.ones((self._n_points, 1))))

        # (n_images, 3, 3)
        Rt = R.transpose(0, 2, 1)

        # (n_images, 3, 3) @ (n_images, 3, 4) -> (n_images, 3, 4)
        P = K @ np.concatenate((Rt, -Rt @ t[..., np.newaxis]), axis=2)

        # (n_images, 3, 4) @ (1, 4, n_points) -> (n_images, 3, n_points) -> (3, n_points, n_images)
        p, q, r = (P @ X_ext.T[np.newaxis]).transpose(1, 2, 0)

        return P, p, q, r

    def _calc_X_diff_pqr(self, P: npt.NDArray) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
        """p, q, rの3次元位置Xに関する微分を求める

        dpdX.shape = (n_points, n_images, 3)
        dqdX.shape = (n_points, n_images, 3)
        drdX.shape = (n_points, n_images, 3)
        """

        # (n_points, n_images, 3)
        dpdX = np.tile(P[:, 0, :3], (self._n_points, 1, 1))
        dqdX = np.tile(P[:, 1, :3], (self._n_points, 1, 1))
        drdX = np.tile(P[:, 2, :3], (self._n_points, 1, 1))

        return dpdX, dqdX, drdX

    def _calc_f_diff_pqr(
        self, p: npt.NDArray, q: npt.NDArray, r: npt.NDArray
    ) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
        """p, q, r焦点距離fに関する微分を求める

        dpdf.shape = (n_points, n_images)
        dqdf.shape = (n_points, n_images)
        drdf.shape = (n_points, n_images)
        """

        # ((n_points, n_images) - (1, n_images) / () * (n_points, n_images)) / (1, n_images)
        # -> (n_points, n_images)
        dpdf = (p - self._u[:, 0][np.newaxis] / self._f0 * r) / self._f[np.newaxis]
        dqdf = (q - self._u[:, 1][np.newaxis] / self._f0 * r) / self._f[np.newaxis]
        drdf = np.zeros(dpdf.shape)

        return dpdf, dqdf, drdf

    def _calc_u_diff_pqr(self, r: npt.NDArray) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
        """p, q, rの光軸点uに関する微分を求める

        dpdu.shape = (n_points, n_images, 2)
        dqdu.shape = (n_points, n_images, 2)
        drdu.shape = (n_points, n_images, 2)
        """

        tmp = r / self._f0
        zero_array = np.zeros(tmp.shape)

        # (n_points, n_images, 2)
        dpdu = np.stack((tmp, zero_array), axis=2)
        dqdu = np.stack((zero_array, tmp), axis=2)
        drdu = np.zeros(dpdu.shape)

        return dpdu, dqdu, drdu

    def _calc_t_diff_pqr(self) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
        """p, q, rの並進tに関する微分を求める

        dpdt.shape = (n_points, n_images, 3)
        dqdt.shape = (n_points, n_images, 3)
        drdt.shape = (n_points, n_images, 3)
        """

        dpdt = np.tile(
            -(self._f[:, np.newaxis] * self._R[:, :, 0] + self._u[:, :1] * self._R[:, :, 2]),
            (self._n_points, 1, 1),
        )
        dqdt = np.tile(
            -(self._f[:, np.newaxis] * self._R[:, :, 1] + self._u[:, -1:] * self._R[:, :, 2]),
            (self._n_points, 1, 1),
        )
        drdt = np.tile(-self._f0 * self._R[:, :, 2], (self._n_points, 1, 1))

        return dpdt, dqdt, drdt

    def _calc_R_diff_pqr(
        self, dpdt: npt.NDArray, dqdt: npt.NDArray, drdt: npt.NDArray
    ) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
        """p, q, rの回転Rに関する微分を求める

        dp_domega.shape = (n_points, n_images, 3)
        dq_domega.shape = (n_points, n_images, 3)
        dr_domega.shape = (n_points, n_images, 3)
        """

        # (n_points, 1, 3) - (1, n_images, 3) -> (n_points, n_images, 3)
        X_minus_t = self._X[:, np.newaxis] - self._t[np.newaxis]

        # (n_points, n_images, 3) x (n_points, n_images, 3) -> (n_points, n_images, 3)
        dp_domega = np.cross(-dpdt, X_minus_t)
        dq_domega = np.cross(-dqdt, X_minus_t)
        dr_domega = np.cross(-drdt, X_minus_t)

        return dp_domega, dq_domega, dr_domega

    def _calc_camera_params_diff_pqr(
        self, p: npt.NDArray, q: npt.NDArray, r: npt.NDArray
    ) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
        """p, q, rのカメラパラメータf,u,v,t,Rに関する微分を取得する

        dp_dparams.shape = (n_points, n_images, 9)
        dq_dparams.shape = (n_points, n_images, 9)
        dr_dparams.shape = (n_points, n_images, 9)
        """

        # (n_points, n_images)
        dpdf, dqdf, drdf = self._calc_f_diff_pqr(p, q, r)

        # (n_points, n_images, 2)
        dpdu, dqdu, drdu = self._calc_u_diff_pqr(r)

        # (n_points, n_images, 3)
        dpdt, dqdt, drdt = self._calc_t_diff_pqr()

        # (n_points, n_images, 3)
        dp_domega, dq_domega, dr_domega = self._calc_R_diff_pqr(dpdt, dqdt, drdt)

        # (n_points, n_images, 9)
        dp_dparams = np.concatenate((dpdf[..., np.newaxis], dpdu, dpdt, dp_domega), axis=2)
        dq_dparams = np.concatenate((dqdf[..., np.newaxis], dqdu, dqdt, dq_domega), axis=2)
        dr_dparams = np.concatenate((drdf[..., np.newaxis], drdu, drdt, dr_domega), axis=2)

        return dp_dparams, dq_dparams, dr_dparams

    def _calc_d_P(
        self,
        p: npt.NDArray,
        q: npt.NDArray,
        r: npt.NDArray,
        dpdX: npt.NDArray,
        dqdX: npt.NDArray,
        drdX: npt.NDArray,
    ) -> npt.NDArray:
        """誤差関数Eの3次元位置Xに関する微分d_Pを求める

        d_P.shape = (3 * n_points, )
        """

        # (n_points, n_images) / (n_points, n_images) - (n_points, n_images) / ()
        # -> (n_points, n_images)
        d1 = p / r - self._x[..., 0] / self._f0

        # (n_points, n_images, 1) * (n_points, n_images, 3)
        # - (n_points, n_images, 1) * (n_points, n_images, 3)
        # -> (n_points, n_images, 3)
        d2 = r[..., np.newaxis] * dpdX - p[..., np.newaxis] * drdX

        # (n_points, n_images) / (n_points, n_images) - (n_points, n_images) / ()
        # -> (n_points, n_images)
        d3 = q / r - self._x[..., 1] / self._f0

        # (n_points, n_images, 1) * (n_points, n_images, 3)
        # - (n_points, n_images, 1) * (n_points, n_images, 3)
        # -> (n_points, n_images, 3)
        d4 = r[..., np.newaxis] * dqdX - q[..., np.newaxis] * drdX

        # (n_points, n_images, 3)
        d = d1[..., np.newaxis] * d2 + d3[..., np.newaxis] * d4
        d = self._visibility_index[..., np.newaxis] * d

        # (n_points, n_images, 1) * (n_points, n_images, 3) / (n_points, n_images, 1)
        # -> (n_points, n_images, 3) -> (n_points, 3) -> (3 * n_points, )
        d_P = 2 * (d / r[..., np.newaxis] ** 2).sum(axis=1).ravel()

        return d_P

    def _calc_d_F(
        self,
        p: npt.NDArray,
        q: npt.NDArray,
        r: npt.NDArray,
        dp_dparams: npt.NDArray,
        dq_dparams: npt.NDArray,
        dr_dparams: npt.NDArray,
    ) -> npt.NDArray:
        """誤差関数Eのカメラパラメータに関する微分d_Fを求める

        d_F.shape = (9 * n_images - 7, )
        """

        # (n_points, n_images) / (n_points, n_images) - (n_points, n_images) / ()
        # -> (n_points, n_images)
        d1 = p / r - self._x[..., 0] / self._f0

        # (n_points, n_images, 1) * (n_points, n_images, 9)
        # - (n_points, n_images, 1) * (n_points, n_images, 9)
        # -> (n_points, n_images, 9)
        d2 = r[..., np.newaxis] * dp_dparams - p[..., np.newaxis] * dr_dparams

        # (n_points, n_images) / (n_points, n_images) - (n_points, n_images) / ()
        # -> (n_points, n_images)
        d3 = q / r - self._x[..., 1] / self._f0

        # (n_points, n_images, 1) * (n_points, n_images, 9)
        # - (n_points, n_images, 1) * (n_points, n_images, 9)
        # -> (n_points, n_images, 9)
        d4 = r[..., np.newaxis] * dq_dparams - q[..., np.newaxis] * dr_dparams

        # (n_points, n_images, 9)
        d = d1[..., np.newaxis] * d2 + d3[..., np.newaxis] * d4
        d = self._visibility_index[..., np.newaxis] * d

        # (n_points, n_images, 1) * (n_points, n_images, 9) / (n_points, n_images, 1)
        # -> (n_points, n_images, 9) -> (n_images, 9) -> (9 * n_images, )
        d_F = 2 * (d / r[..., np.newaxis] ** 2).sum(axis=0).ravel()

        included_ind = np.ones(d_F.shape, dtype=np.bool_)
        included_ind[self._remove_ind] = False

        # (9 * n_images - 7, )
        d_F = d_F[included_ind]

        return d_F

    def _calc_matE(
        self,
        p: npt.NDArray,
        q: npt.NDArray,
        r: npt.NDArray,
        dpdX: npt.NDArray,
        dqdX: npt.NDArray,
        drdX: npt.NDArray,
    ) -> npt.NDArray:
        """誤差関数Eの3次元位置Xに関する2回微分matEを求める

        matE.shape = (n_points, 3, 3)
        """

        # (n_points, n_images, 1) * (n_points, n_images, 3)
        # - (n_points, n_images, 1) * (n_points, n_images, 3)
        # -> (n_points, n_images, 3)
        d1 = r[..., np.newaxis] * dpdX - p[..., np.newaxis] * drdX

        # (n_points, n_images, 1) * (n_points, n_images, 3)
        # - (n_points, n_images, 1) * (n_points, n_images, 3)
        # -> (n_points, n_images, 3)
        d2 = r[..., np.newaxis] * dqdX - q[..., np.newaxis] * drdX

        # (n_points, n_images, 3, 1) @ (n_points, n_images, 1, 3)
        # + (n_points, n_images, 3, 1) @ (n_points, n_images, 1, 3)
        # -> (n_points, n_images, 3, 3)
        d = (
            d1[..., np.newaxis] @ d1[..., np.newaxis, :]
            + d2[..., np.newaxis] @ d2[..., np.newaxis, :]
        )
        d = self._visibility_index[..., np.newaxis, np.newaxis] * d

        # (n_points, n_images, 1, 1) * (n_points, n_images, 3, 3) / (n_points, n_images, 1, 1)
        # -> (n_points, n_images, 3, 3) -> (n_points, 3, 3)
        matE = 2 * (d / r[..., np.newaxis, np.newaxis] ** 4).sum(axis=1)

        return matE

    def _calc_matF(
        self,
        p: npt.NDArray,
        q: npt.NDArray,
        r: npt.NDArray,
        dpdX: npt.NDArray,
        dqdX: npt.NDArray,
        drdX: npt.NDArray,
        dp_dparams: npt.NDArray,
        dq_dparams: npt.NDArray,
        dr_dparams: npt.NDArray,
    ) -> npt.NDArray:
        """誤差関数Eの3次元位置Xとカメラパラメータに関する2回微分matFを求める

        matF.shape = (n_points, 3, 9 * n_images - 7)
        """

        # (n_points, n_images, 1) * (n_points, n_images, 3)
        # - (n_points, n_images, 1) * (n_points, n_images, 3)
        # -> (n_points, n_images, 3)
        d1 = r[..., np.newaxis] * dpdX - p[..., np.newaxis] * drdX

        # (n_points, n_images, 1) * (n_points, n_images, 9)
        # - (n_points, n_images, 1) * (n_points, n_images, 9)
        # -> (n_points, n_images, 9)
        d2 = r[..., np.newaxis] * dp_dparams - p[..., np.newaxis] * dr_dparams

        # (n_points, n_images, 1) * (n_points, n_images, 3)
        # - (n_points, n_images, 1) * (n_points, n_images, 3)
        # -> (n_points, n_images, 3)
        d3 = r[..., np.newaxis] * dqdX - q[..., np.newaxis] * drdX

        # (n_points, n_images, 1) * (n_points, n_images, 9)
        # - (n_points, n_images, 1) * (n_points, n_images, 9)
        # -> (n_points, n_images, 9)
        d4 = r[..., np.newaxis] * dq_dparams - q[..., np.newaxis] * dr_dparams

        # (n_points, n_images, 3, 1) @ (n_points, n_images, 1, 9)
        # + (n_points, n_images, 3, 1) @ (n_points, n_images, 1, 9)
        # -> (n_points, n_images, 3, 9)
        d = (
            d1[..., np.newaxis] @ d2[..., np.newaxis, :]
            + d3[..., np.newaxis] @ d4[..., np.newaxis, :]
        )
        d = self._visibility_index[..., np.newaxis, np.newaxis] * d

        # (n_points, n_images, 3, 9) / (n_points, n_images, 1, 1) -> (n_points, n_images, 3, 9)
        matF = 2 * d / r[..., np.newaxis, np.newaxis] ** 4

        # (n_points, n_images, 3, 9) -> (n_points, 3, n_images, 9) -> (n_points, 3, 9 * n_images)
        matF = matF.transpose(0, 2, 1, 3).reshape(self._n_points, 3, -1)

        included_ind = np.ones(matF.shape[2], dtype=np.bool_)
        included_ind[self._remove_ind] = False

        # (n_points, 3, 9 * n_images - 7)
        matF = matF[:, :, included_ind]

        return matF

    def _calc_matG(
        self,
        p: npt.NDArray,
        q: npt.NDArray,
        r: npt.NDArray,
        dp_dparams: npt.NDArray,
        dq_dparams: npt.NDArray,
        dr_dparams: npt.NDArray,
    ) -> npt.NDArray:
        """誤差関数Eのカメラパラメータに関する2回微分matGを求める

        matG.shape = (9 * n_images - 7, 9 * n_images - 7)
        """

        # (n_points, n_images, 1) * (n_points, n_images, 9)
        # - (n_points, n_images, 1) * (n_points, n_images, 9)
        # -> (n_points, n_images, 9)
        d1 = r[..., np.newaxis] * dp_dparams - p[..., np.newaxis] * dr_dparams

        # (n_points, n_images, 1) * (n_points, n_images, 9)
        # - (n_points, n_images, 1) * (n_points, n_images, 9)
        # -> (n_points, n_images, 9)
        d2 = r[..., np.newaxis] * dq_dparams - q[..., np.newaxis] * dr_dparams

        # (n_points, n_images, 9, 1) @ (n_points, n_images, 1, 9)
        # + (n_points, n_images, 9, 1) @ (n_points, n_images, 1, 9)
        # -> (n_points, n_images, 9, 9)
        d = (
            d1[..., np.newaxis] @ d1[..., np.newaxis, :]
            + d2[..., np.newaxis] @ d2[..., np.newaxis, :]
        )
        d = self._visibility_index[..., np.newaxis, np.newaxis] * d

        # (n_points, n_images, 9, 9) / (n_points, n_images, 1, 1)
        # -> (n_points, n_images, 9, 9) -> (n_images, 9, 9)
        matG = 2 * (d / r[..., np.newaxis, np.newaxis] ** 4).sum(axis=0)

        # (9 * n_images, 9 * n_images)
        matG = block_diag(*matG)

        included_ind = np.ones(matG.shape[0], dtype=np.bool_)
        included_ind[self._remove_ind] = False

        # (9 * n_images - 7, 9 * n_images - 7)
        matG = matG[included_ind][:, included_ind]

        return matG

    def _calc_reprojection_error(self, p: npt.NDArray, q: npt.NDArray, r: npt.NDArray) -> float:
        """再投影誤差Eを求める"""

        # (n_points, n_images)
        x1 = self._x[:, :, 0]
        x2 = self._x[:, :, 1]

        E = (
            self._visibility_index * ((p / r - x1 / self._f0) ** 2 + (q / r - x2 / self._f0) ** 2)
        ).sum()

        return E
