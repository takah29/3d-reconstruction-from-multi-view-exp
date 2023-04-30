import numpy as np

from .utils import unit_vec


class Camera:
    def __init__(self, origin=(0.0, 0.0, 0.0), target=(0.0, 0.0, 1.0), f=1.0):
        origin = np.asarray(origin)
        target = np.asarray(target)
        self.o = origin
        self.d = unit_vec(target - origin)
        self.f = f

    def get_camera_matrix(self, f0):
        return self._calc_camera_matrix(f0)

    def get_pose(self):
        return self._calc_pose()

    def project_points(self, X, f0, method="perspective"):
        """3次元点を画像面に投影する"""
        if method == "perspective":
            res = self._perspective_projection(X, f0)
        elif method == "orthographic":
            res = self._orthographic_projection(X)
        else:
            raise ValueError()

        return res

    def _calc_camera_matrix(self, f0):
        R, t = self._calc_pose()
        return np.diag((self.f, self.f, f0)) @ np.hstack([R.T, -R.T @ t[:, np.newaxis]])

    def _calc_pose(self):
        world_top = np.array([1.0, 0.0, 0.0])
        camera_z = self.d
        camera_y = np.cross(camera_z, world_top)  # camera right
        camera_x = np.cross(camera_y, camera_z)  # camera up
        R = np.vstack((camera_x, camera_y, camera_z)).T
        t = self.o
        return R, t

    def _perspective_projection(self, X, f0):
        X_ext = np.hstack((X, np.ones((X.shape[0], 1))))
        Xproj = X_ext @ self.get_camera_matrix(f0).T
        return Xproj[:, :2] / Xproj[:, -1:]

    def _orthographic_projection(self, X):
        X_ext = np.hstack((X, np.ones((X.shape[0], 1))))
        R, t = self._calc_pose()
        Xproj = X_ext @ np.hstack([R.T, -R.T @ t[:, np.newaxis]]).T
        return Xproj[:, :2]


if __name__ == "__main__":
    # 簡易テスト
    import numpy.testing as nptest

    X = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])

    # case1
    camera = Camera((0, 0, -1), (0, 0, 1), f=1)
    Xproj = camera.project_points(X, 1.0)
    nptest.assert_array_almost_equal(Xproj, np.array([[0, 0], [1, 0], [0, 1], [0, 0]]))

    # case2
    camera = Camera((0, -1, 0), (0, 1, 0), f=1)
    Xproj = camera.project_points(X, 1.0)
    nptest.assert_array_almost_equal(Xproj, np.array([[0, 0], [1, 0], [0, 0], [0, -1]]))

    print("Passed all tests.")
