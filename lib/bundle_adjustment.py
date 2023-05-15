from .perspective_camera_calibration import _normalize_world_axis_with_first_camera
from .utils import get_rotation_matrix

import numpy as np
from scipy.linalg import block_diag


class BundleAdjuster:
    def __init__(self, x_list, init_X, init_K, init_R, init_t, f0=1.0):
        # (n_points, n_images, 2)
        self._x = np.stack(x_list, axis=1)

        # (n_points, 3), (n_images, 3, 3), (n_images, 3)
        self._X, self._R, self._t = _normalize_world_axis_with_first_camera(init_X, init_R, init_t)

        # (n_images, )
        self._f = init_K[:, 0, 0]

        # (n_images, 2)
        self._u = init_K[:, :2, 2]

        self._f0 = f0

        self._n_points = init_X.shape[0]
        self._n_images = init_R.shape[0]

        # カメラパラメータはR1=I, t1=0, t2_2=1を仮定しているので、該当の未知数を削除するためのインデックスを用意する
        # カメラパラメータ: (f1, u01, v01, t1_1, t1_2, t1_3, omega1_1, omega1_2, omega1_3, f2, u02, ...)
        self._excluded_ind = np.array([3, 4, 5, 6, 7, 8, 13])

    def optimize(self, tol):
        """再投影誤差を最小化するX, K, R, tを求める"""
        c = 0.0001

        while True:
            K = self._get_K(self._f, self._u)
            P, p, q, r = self._calc_pqr(self._X, K, self._R, self._t)

            E = self._calc_reprojection_error(p, q, r)

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
                matGc = matG.copy()
                tmp_ind = np.arange(3)
                matEc[:, tmp_ind, tmp_ind] *= 1 + c
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

                # (n_points, 9 * n_images - 7, 3) @ (n_points, 3, 1) -> (n_points, 9 * n_images - 7, 1)
                # -> (n_points, 9 * n_images - 7) -> (9 * n_images - 7, )
                b = (FtEinv @ delta_X_E).squeeze().sum(axis=0) - d_F

                # (9 * n_images - 7, )
                delta_xi_F = np.linalg.solve(A, b)

                # (n_points, 3, 3) @
                # ((n_points, 3, 9 * n_images - 7) @ (9 * n_images - 7, 1) + (n_points, 3, 1))
                # + (n_points, 3, 1)
                # -> (n_points, 3, 1)
                # -> (n_points, 3)
                delta_X = -(
                    matEinv @ (matF @ delta_xi_F[:, np.newaxis] + delta_X_E)
                ).squeeze()

                # パラメータの更新
                tmp_X = self._update_X(delta_X)
                tmp_f, tmp_u, tmp_t, tmp_R = self._update_camera_params(delta_xi_F)

                # 更新したパラメータで再投影誤差を計算する
                tmp_K = self._get_K(tmp_f, tmp_u)
                _, tmp_p, tmp_q, tmp_r = self._calc_pqr(tmp_X, tmp_K, tmp_R, tmp_t)
                E_ = self._calc_reprojection_error(tmp_p, tmp_q, tmp_r)
                print(E, E_)

                if E_ > E:
                    c *= 10
                else:
                    break

            self._X = tmp_X
            self._f = tmp_f
            self._u = tmp_u
            self._t = tmp_t
            self._R = tmp_R

            # 再投影誤差が変化しなくなったら終了
            if np.abs(E_ - E) <= tol:
                break
            else:
                E = E_
                c /= 10

        return self._X, self._get_K(self._f, self._u), self._R, self._t

    def _update_X(self, delta_X):
        return self._X + delta_X

    def _update_camera_params(self, delta_xi_F):
        # (9 * n_images, )
        delta_xi_F = np.insert(delta_xi_F, self._excluded_ind, np.zeros(self._excluded_ind.shape))

        # (n_images, 9)
        delta_xi_F = delta_xi_F.reshape(self._n_images, 9)

        # (n_images, 1), (n_images, 2), (n_images, 3), (n_images, 3)
        delta_f, delta_u, delta_t, delta_omega = np.hsplit(delta_xi_F, [1, 3, 6])

        # (n_images, )
        delta_f = delta_f.squeeze()

        # (n_images, 3, 3)
        delta_R = np.stack([get_rotation_matrix(do) for do in delta_omega])

        return self._f + delta_f, self._u + delta_u, self._t + delta_t, delta_R @ self._R

    def _get_K(self, f, u):
        K = np.zeros((self._n_images, 3, 3))
        K[:, (0, 1), (0, 1)] = f[:, np.newaxis]
        K[:, :2, 2] = u
        K[:, 2, 2] = self._f0

        return K

    def _calc_pqr(self, X, K, R, t):
        # (n_points, 4)
        X_ext = np.hstack((X, np.ones((self._n_points, 1))))

        # (n_images, 3, 3)
        Rt = R.transpose(0, 2, 1)

        # (n_images, 3, 3) @ (n_images, 3, 4) -> (n_images, 3, 4)
        P = K @ np.concatenate((Rt, -Rt @ t[..., np.newaxis]), axis=2)

        # (n_images, 3, 4) @ (4, n_points) -> (n_images, 3, n_points) -> (3, n_points, n_images)
        p, q, r = (P @ X_ext.T[np.newaxis]).transpose(1, 2, 0)

        return P, p, q, r

    def _calc_X_diff_pqr(self, P):
        """p, q, rの3次元位置Xに関する微分を求める

        dpdX.shape = (n_points, n_images, 3)
        dqdX.shape = (n_points, n_images, 3)
        drdX.shape = (n_points, n_images, 3)
        """
        dpdX = []
        dqdX = []
        drdX = []
        for _ in range(self._n_points):
            # P.shape = (n_images, 3, 4)
            dpdX.append(P[:, 0, :3])
            dqdX.append(P[:, 1, :3])
            drdX.append(P[:, 2, :3])

        # (n_points, n_images, 3)
        dpdX = np.stack(dpdX)
        dqdX = np.stack(dqdX)
        drdX = np.stack(drdX)

        return dpdX, dqdX, drdX

    def _calc_f_diff_pqr(self, p, q, r):
        """p, q, r焦点距離fに関する微分を求める"""
        # ((n_points, n_images) - (1, n_images) / () * (n_points, n_images)) / (1, n_images)
        # -> (n_points, n_images)
        dpdf = (p - self._u[:, 0][np.newaxis] / self._f0 * r) / self._f[np.newaxis]
        dqdf = (q - self._u[:, 1][np.newaxis] / self._f0 * r) / self._f[np.newaxis]
        drdf = np.zeros(dpdf.shape)

        return dpdf, dqdf, drdf

    def _calc_u_diff_pqr(self, r):
        """p, q, rの光軸点uに関する微分を求める"""
        tmp = r / self._f0
        zero_array = np.zeros(tmp.shape)

        # (n_points, n_images, 2)
        dpdu = np.stack((tmp, zero_array), axis=2)
        dqdu = np.stack((np.zeros(tmp.shape), tmp), axis=2)
        drdu = np.zeros(dpdu.shape)

        return dpdu, dqdu, drdu

    def _calc_t_diff_pqr(self):
        """p, q, rの並進tに関する微分を求める"""
        dpdt = []
        dqdt = []
        drdt = []
        for _ in range(self._n_points):
            # (n_images, 1) * (n_images, 3) + (n_images, 1) * (n_images, 3) -> (n_images, 3)
            dpdt.append(
                self._f[..., np.newaxis] * self._R[:, :, 0] + self._u[:, :1] * self._R[:, :, 2]
            )
            dqdt.append(
                self._f[..., np.newaxis] * self._R[:, :, 1] + self._u[:, -1:] * self._R[:, :, 2]
            )

            # () * (n_images, 3) -> (n_images, 3)
            drdt.append(self._f0 * self._R[:, :, 2])

        # (n_points, n_images, 3)
        dpdt = np.stack(dpdt)
        dqdt = np.stack(dqdt)
        drdt = np.stack(drdt)

        return dpdt, dqdt, drdt

    def _calc_R_diff_pqr(self, dpdt, dqdt, drdt):
        """p, q, rの回転Rに関する微分を求める"""
        # (n_points, 1, 3) - (1, n_images, 3) -> (n_points, n_images, 3)
        X_minus_t = self._X[:, np.newaxis] - self._t[np.newaxis]

        # (n_points, n_images, 3) x (n_points, n_images, 3) -> (n_points, n_images, 3)
        dp_domega = np.cross(dpdt, X_minus_t)
        dq_domega = np.cross(dqdt, X_minus_t)
        dr_domega = np.cross(drdt, X_minus_t)

        return dp_domega, dq_domega, dr_domega

    def _calc_camera_params_diff_pqr(self, p, q, r):
        """p, q, rのカメラパラメータf,u,v,t,Rに関する微分を取得する"""
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

    def _calc_d_P(self, p, q, r, dpdX, dqdX, drdX):
        """誤差関数Eの3次元位置Xに関する微分d_Pを求める"""
        # (n_points, n_images) / (n_points, n_images) - (n_points, n_images) / ()
        # -> (n_points, n_images)
        de1 = p / r - self._x[..., 0] / self._f0

        # (n_points, n_images, 1) * (n_points, n_images, 3)
        # - (n_points, n_images, 1) * (n_points, n_images, 3)
        # -> (n_points, n_images, 3)
        de2 = r[..., np.newaxis] * dpdX - p[..., np.newaxis] * drdX

        # (n_points, n_images) / (n_points, n_images) - (n_points, n_images) / ()
        # -> (n_points, n_images)
        de3 = q / r - self._x[..., 1] / self._f0

        # (n_points, n_images, 1) * (n_points, n_images, 3)
        # - (n_points, n_images, 1) * (n_points, n_images, 3)
        # -> (n_points, n_images, 3)
        de4 = r[..., np.newaxis] * dqdX - q[..., np.newaxis] * drdX

        # (n_points, n_images, 3)
        de = de1[..., np.newaxis] * de2 + de3[..., np.newaxis] * de4

        # (n_points, n_images, 3) / (n_points, n_images, 1)
        # -> (n_points, n_images, 3) -> (n_points, 3) -> (3 * n_points, )
        d_P = 2 * (de / (r**2)[..., np.newaxis]).sum(axis=1).ravel()

        return d_P

    def _calc_d_F(self, p, q, r, dp_dparams, dq_dparams, dr_dparams):
        """誤差関数Eのカメラパラメータに関する微分d_Fを求める"""
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

        # (n_points, n_images, 9) / (n_points, n_images, 1)
        # -> (n_points, n_images, 9) -> (n_images, 9) -> (9 * n_images, )
        d_F = 2 * (d / r[..., np.newaxis] ** 2).sum(axis=0).ravel()

        included_arr = np.ones(d_F.shape, dtype=np.bool_)
        included_arr[self._excluded_ind] = False

        # (9 * n_images - 7, )
        d_F = d_F[included_arr]

        return d_F

    def _calc_matE(self, p, q, r, dpdX, dqdX, drdX):
        """誤差関数Eの3次元位置Xに関する2回微分matEを求める"""
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

        # (n_points, n_images, 3, 3) / (n_points, n_images, 1, 1)
        # -> (n_points, n_images, 3, 3) -> (n_points, 3, 3)
        matE = 2 * (d / r[..., np.newaxis, np.newaxis] ** 4).sum(axis=1)

        return matE

    def _calc_matF(self, p, q, r, dpdX, dqdX, drdX, dp_dparams, dq_dparams, dr_dparams):
        """誤差関数Eの3次元位置Xとカメラパラメータに関する2回微分matFを求める"""
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

        # (n_points, n_images, 3, 9) / (n_points, n_images, 1, 1) -> (n_points, n_images, 3, 9)
        matF = 2 * d / r[..., np.newaxis, np.newaxis] ** 4

        # (n_points, n_images, 3, 9) -> (n_points, 3, n_images, 9) -> (n_points, 3, 9 * n_images)
        matF = matF.transpose(0, 2, 1, 3).reshape(self._n_points, 3, -1)

        included_arr = np.ones(matF.shape[2], dtype=np.bool_)
        included_arr[self._excluded_ind] = False

        # (n_points, 3, 9 * n_images - 7)
        matF = matF[:, :, included_arr]

        return matF

    def _calc_matG(self, p, q, r, dp_dparams, dq_dparams, dr_dparams):
        """誤差関数Eのカメラパラメータに関する2回微分matGを求める"""
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

        # (n_points, n_images, 9, 9) / (n_points, n_images, 1, 1)
        # -> (n_points, n_images, 9, 9) -> (n_images, 9, 9)
        matG = 2 * (d / r[..., np.newaxis, np.newaxis] ** 4).sum(axis=0)

        # (9 * n_images, 9 * n_images)
        matG = block_diag(*matG)

        included_arr = np.ones(matG.shape[0], dtype=np.bool_)
        included_arr[self._excluded_ind] = False

        # (9 * n_images - 7, 9 * n_images - 7)
        matG = matG[included_arr][:, included_arr]

        return matG

    def _calc_reprojection_error(self, p, q, r):
        """再投影誤差Eを求める"""
        # (n_images, n_points)
        x1 = self._x[:, :, 0]
        x2 = self._x[:, :, 1]

        E = ((p / r - x1 / self._f0) ** 2 + (q / r - x2 / self._f0) ** 2).sum()

        return E
