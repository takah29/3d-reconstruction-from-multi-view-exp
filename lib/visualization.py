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

    def set_lim(self, xlim=[-5.0, 5.0], ylim=[-5.0, 5.0], zlim=[-5.0, 5.0]):
        self.ax.set_xlim3d(ylim)
        self.ax.set_ylim3d(zlim)
        self.ax.set_zlim3d(xlim)

    def plot_basis(self, basis: NDArray, pos: NDArray, label=None) -> None:
        """
        基底をプロットする。回転行列やカメラの姿勢のプロットにも使用できる。
        """
        assert pos.shape == (3,)
        assert basis.shape == (3, 3)

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

    def plot_points(self, X: NDArray, color="black") -> None:
        """3次元点群をプロットする、colorはリストで与えても良い"""
        self.ax.scatter(X[:, 1], X[:, 2], X[:, 0], c=color, marker="o")

    def show(self):
        """3次元グラフを表示する"""
        plt.show()

    def clear(self):
        plt.clf()


def plot_2d_points(x, ax, color="black") -> None:
    """2次元点群をプロットする、colorはリストで与えても良い"""
    ax.scatter(x[:, 1], x[:, 0], c=color, marker="o")


def subplot_2d_points(x1, x2, color="black"):
    ax1 = plt.subplot(1, 2, 1)
    ax1.set_xlim(-2, 2)
    ax1.set_ylim(-2, 2)
    plt.grid()
    plot_2d_points(x1, ax1, color=color)

    ax2 = plt.subplot(1, 2, 2)
    ax2.set_xlim(-2, 2)
    ax2.set_ylim(-2, 2)
    plt.grid()
    plot_2d_points(x2, ax2, color=color)


if __name__ == "__main__":
    import numpy as np

    ax = init_3d_ax()

    # 基底のプロット
    pos = np.array([1, 0, 0])
    omega = 1.0
    basis = np.array(
        [[1, 0, 0], [0, np.cos(omega), -np.sin(omega)], [0, np.sin(omega), np.cos(omega)]]
    ).T  # 横ベクトル向けに転置する
    plot_3d_basis(pos, basis, ax, label="test")

    # データ点のプロット
    X = np.eye(3) @ basis + pos
    plot_3d_points(X, ax, "blue")
    plt.show()
