import numpy as np
import numpy.typing as npt

from .utils import unit_vec


class Camera:
    def __init__(self, R: npt.NDArray, t: npt.NDArray, K: npt.NDArray = np.eye(3)):
        self._R = R
        self._t = t
        self._K = K

    def get_camera_matrix(self) -> npt.NDArray:
        return self._K @ np.hstack([self._R.T, -self._R.T @ self._t[:, np.newaxis]])

    def get_parameters(self) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
        return self._K, self._R, self._t

    def project_points(self, X: npt.NDArray, method: str = "perspective") -> npt.NDArray:
        """3次元点を画像面に投影する"""
        if method == "perspective":
            res = self._perspective_projection(X)
        elif method == "orthographic":
            res = self._orthographic_projection(X)
        else:
            raise ValueError()

        return res

    def _perspective_projection(self, X: npt.NDArray) -> npt.NDArray:
        X_ext = np.hstack((X, np.ones((X.shape[0], 1))))
        Xproj = X_ext @ self.get_camera_matrix().T

        return Xproj[:, :2] / Xproj[:, -1:]

    def _orthographic_projection(self, X: npt.NDArray) -> npt.NDArray:
        X_ext = np.hstack((X, np.ones((X.shape[0], 1))))
        _, R, t = self.get_parameters()
        X_ext_ = X_ext @ np.hstack([R.T, -R.T @ t[:, np.newaxis]]).T

        return X_ext_[:, :2]

    @staticmethod
    def _calc_external_params(
        origin: npt.NDArray,
        direction: npt.NDArray,
    ) -> tuple[npt.NDArray, npt.NDArray]:
        world_top = np.array([1.0, 0.0, 0.0])
        camera_z = direction
        camera_y = unit_vec(np.cross(camera_z, world_top))  # camera right
        camera_x = unit_vec(np.cross(camera_y, camera_z))  # camera up
        R = np.vstack((camera_x, camera_y, camera_z)).T
        t = origin

        return R, t

    @staticmethod
    def create(
        origin: npt.NDArray | tuple[float, float, float] = (0.0, 0.0, 0.0),
        target: npt.NDArray | tuple[float, float, float] = (0.0, 0.0, 1.0),
        f: float = 1.0,
        f0: float = 1.0,
    ) -> "Camera":
        origin = np.asarray(origin)
        target = np.asarray(target)
        direction = unit_vec(target - origin)

        R, t = Camera._calc_external_params(origin, direction)
        K = np.diag((f, f, f0))

        return Camera(R, t, K)


def calc_projected_points(X, K, R, t):
    """すべてのカメラで投影点を求める"""
    x_list = []
    for R_pred, t_pred, K_pred in zip(R, t, K):
        x = Camera(R_pred, t_pred, K_pred).project_points(X, method="perspective")
        x_list.append(x)

    return x_list


def get_camera_parames(camera_list):
    """カメラの内部パラメータと外部パラメータを取得する"""
    K = []
    R = []
    t = []
    for camera in camera_list:
        K_, R_, t_ = camera.get_parameters()
        K.append(K_)
        R.append(R_)
        t.append(t_)
    K = np.stack(K)
    R = np.stack(R)
    t = np.stack(t)

    return K, R, t


if __name__ == "__main__":
    # 簡易テスト
    import numpy.testing as nptest

    X = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])

    # case1
    camera = Camera.create((0, 0, -1), (0, 0, 1), f=1)
    Xproj = camera.project_points(X)
    nptest.assert_array_almost_equal(Xproj, np.array([[0, 0], [1, 0], [0, 1], [0, 0]]))

    # case2
    camera = Camera.create((0, -1, 0), (0, 1, 0), f=1)
    Xproj = camera.project_points(X)
    nptest.assert_array_almost_equal(Xproj, np.array([[0, 0], [1, 0], [0, 0], [0, -1]]))

    print("Passed all tests.")
