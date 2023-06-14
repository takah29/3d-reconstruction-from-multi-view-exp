import matplotlib.pyplot as plt
from numpy.typing import NDArray


class ThreeDimensionalPlotter:
    def __init__(self, figsize: tuple[int, int] | None = None, title: str | None = None):
        """Xを上、Zを奥行きとした右手系座標を設定する"""
        self.fig = plt.figure(figsize=figsize)

        self.ax = plt.axes(projection="3d")
        self.ax.set_title(title)
        self.ax.set_xlabel("Y")
        self.ax.set_ylabel("Z")
        self.ax.set_zlabel("X")
        self.ax.set_box_aspect((1, 1, 1))

    def set_lim(
        self,
        xlim: list[float] = [-5.0, 5.0],
        ylim: list[float] = [-5.0, 5.0],
        zlim: list[float] = [-5.0, 5.0],
    ) -> None:
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

    def plot_points(self, X: NDArray, color: str | list | None = "black") -> None:
        """3次元点群をプロットする、colorはリストで与えても良い"""
        self.ax.scatter(X[:, 1], X[:, 2], X[:, 0], c=color, marker=".")

    def show(self) -> None:
        """3次元グラフを表示する"""
        plt.show()

    def close(self) -> None:
        plt.clf()
        plt.close()

    def pause(self, s=0.1) -> None:
        plt.pause(s)


class TwoDimensionalMatrixPlotter:
    def __init__(
        self, n_row: int, n_col: int, figsize: tuple[int, int] | None = None, is_grid: bool = True
    ) -> None:
        plt.figure(figsize=figsize)

        self.n_row = n_row
        self.n_col = n_col
        self.is_grid = is_grid

    def select(self, plot_id: int) -> None:
        self.current_ax = plt.subplot(self.n_row, self.n_col, plot_id + 1)

    def set_property(self, title: str, xlim=[-1.0, 1.0], ylim=[-1.0, 1.0]) -> None:
        self.current_ax.set_title(title)
        self.current_ax.set_aspect("equal")

        # matplotlibと違いx上、y右を想定しているので変数名が異なっている
        self.current_ax.set_xlim(ylim)
        self.current_ax.set_ylim(xlim)

        if self.is_grid:
            self.current_ax.grid()

    def plot_points(self, x: NDArray, color="black", label=None) -> None:
        """2次元点群をプロットする、colorはリストで与えても良い"""
        self.current_ax.scatter(x[:, 1], x[:, 0], c=color, marker=".", label=label)
        if label is not None:
            self.current_ax.legend()

    def show(self) -> None:
        """2次元グラフを表示する"""
        plt.show()

    def close(self) -> None:
        plt.clf()
        plt.close()


def show_3d_scene_data(X: NDArray, R: NDArray, t: NDArray, color: str | list | None = None) -> None:
    """データ点とカメラの姿勢を3Dプロットして表示する"""
    plotter_3d = ThreeDimensionalPlotter(figsize=(10, 10))
    plotter_3d.set_lim()
    plotter_3d.plot_points(X, color=color)
    for i, (R_, t_) in enumerate(zip(R, t), start=1):
        plotter_3d.plot_basis(R_, t_, label=f"Camera{i}")
    plotter_3d.show()
    plotter_3d.close()


def show_2d_projection_data(
    x_list: list[NDArray],
    reproj_x_list: list[NDArray] | None = None,
    n_col: int = 6,
    xlim=(-0.5, 0.5),
    ylim=(-0.5, 0.5),
) -> None:
    """投影点と再投影点をプロットして表示する"""
    n_images = len(x_list)
    n_row = (n_images - 1) // n_col + 1
    plotter_2d = TwoDimensionalMatrixPlotter(n_row, n_col, (20, 6))
    for i in range(n_row):
        range_width = range(n_images % n_col) if i == n_images // n_col else range(n_col)
        for j in range_width:
            # camera(i * j)で射影した2次元データ点のプロット
            plotter_2d.select(n_col * i + j)
            plotter_2d.set_property(f"Camera {n_col * i + j + 1}", xlim, ylim)

            plotter_2d.plot_points(x_list[n_col * i + j], color="green", label="Projection")

            if reproj_x_list is not None:
                plotter_2d.plot_points(
                    reproj_x_list[n_col * i + j], color="red", label="Reprojection"
                )

    plotter_2d.show()
    plotter_2d.close()


def animate(data: list[dict[str, NDArray]]):
    """基底と点群をアニメーションでプロットする

    dataは以下のデータ構造を仮定する
    data = [
        {"points": X1, "basis": R1, "pos": t1}
        {"points": X2, "basis": R2, "pos": t2}
        ...
        {"points": Xn, "basis": Rn, "pos": tn}
    ]
    """
    plotter_3d = ThreeDimensionalPlotter()

    while plt.fignum_exists(plotter_3d.fig.number):
        for d in data:
            X = d["points"]
            R = d["basis"]
            t = d["pos"]

            plotter_3d.set_lim()
            plotter_3d.plot_points(X)
            for i, (R_, t_) in enumerate(zip(R, t), start=1):
                plotter_3d.plot_basis(R_, t_, label=f"camera{i}")

            plotter_3d.pause(0.05)
            plotter_3d.ax.cla()


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
