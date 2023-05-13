import matplotlib.pyplot as plt
from numpy.typing import NDArray


class ThreeDimensionalPlotter:
    def __init__(self, figsize=None, title=None):
        """Xを上、Zを奥行きとした右手系座標を設定する"""
        plt.figure(figsize=figsize)

        self.ax = plt.axes(projection="3d")
        self.ax.set_title(title)
        self.ax.set_xlabel("Y")
        self.ax.set_ylabel("Z")
        self.ax.set_zlabel("X")
        self.ax.set_box_aspect((1, 1, 1))

    def set_lim(self, xlim=[-5.0, 5.0], ylim=[-5.0, 5.0], zlim=[-5.0, 5.0]):
        self.ax.set_xlim3d(ylim)
        self.ax.set_ylim3d(zlim)
        self.ax.set_zlim3d(xlim)

    def plot_basis(self, basis: NDArray, pos: NDArray, label: str | None = None) -> None:
        """
        基底をプロットする。回転行列やカメラの姿勢のプロットにも使用できる。
        """
        assert basis.shape == (3, 3)
        assert pos.shape == (3,)

        basis = basis.T

        cols = ["r", "g", "b", "r", "r", "g", "g", "b", "b"]
        _ = self.ax.quiver(
            [pos[1]] * 3,
            [pos[2]] * 3,
            [pos[0]] * 3,
            basis[:, 1],
            basis[:, 2],
            basis[:, 0],
            colors=cols,
        )

        if label is not None:
            self.ax.text(pos[1], pos[2], pos[0], label)

    def plot_points(self, X: NDArray, color: str | list = "black") -> None:
        """3次元点群をプロットする、colorはリストで与えても良い"""
        self.ax.scatter(X[:, 1], X[:, 2], X[:, 0], c=color, marker="o")

    def show(self):
        """3次元グラフを表示する"""
        plt.show()

    def close(self):
        plt.clf()
        plt.close()


class TwoDimensionalMatrixPlotter:
    def __init__(self, n_row, n_col, figsize=None, is_grid=True):
        plt.figure(figsize=figsize)

        self.n_row = n_row
        self.n_col = n_col
        self.is_grid = is_grid

    def select(self, plot_id: int):
        self.current_ax = plt.subplot(self.n_row, self.n_col, plot_id + 1)

    def set_property(self, title, xlim=[-1.0, 1.0], ylim=[-1.0, 1.0]):
        self.current_ax.set_title(title)
        self.current_ax.set_aspect("equal")
        self.current_ax.set_xlim(ylim)
        self.current_ax.set_ylim(xlim)
        if self.is_grid:
            self.current_ax.grid()

    def plot_points(self, x: NDArray, color="black", label=None) -> None:
        """2次元点群をプロットする、colorはリストで与えても良い"""
        self.current_ax.scatter(x[:, 1], x[:, 0], c=color, marker=".", label=label)
        if label is not None:
            self.current_ax.legend()

    def show(self):
        """2次元グラフを表示する"""
        plt.show()

    def close(self):
        plt.clf()
        plt.close()


if __name__ == "__main__":
    import numpy as np

    plotter_3d = ThreeDimensionalPlotter()
    plotter_3d.set_lim([-2, 2], [-2, 2], [-2, 2])

    # 基底のプロット
    pos = np.array([0, 1, 0])
    omega = 1.0
    basis = np.array(
        [[1, 0, 0], [0, np.cos(omega), -np.sin(omega)], [0, np.sin(omega), np.cos(omega)]]
    )  # 横ベクトル向けに転置する
    plotter_3d.plot_basis(basis, pos, label="test")

    # データ点のプロット
    X = np.eye(3) @ basis.T + pos
    plotter_3d.plot_points(X, "blue")
    plotter_3d.show()
    plotter_3d.close()
