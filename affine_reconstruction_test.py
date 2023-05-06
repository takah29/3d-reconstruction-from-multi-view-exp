import matplotlib.pyplot as plt
import numpy as np

from lib.camera import Camera
from lib.affine_camera_calibration import (
    orthographic_self_calibration,
    symmetric_affine_self_calibration,
    paraperspective_self_calibration,
)
from lib.utils import sample_hemisphere_points, set_points1
from lib.visualization import ThreeDimensionalPlotter, plot_2d_points


def main():
    np.random.seed(123)

    f = 1.0
    image_num = 6

    # カメラの設定
    camera_pos = sample_hemisphere_points(image_num, 5)
    targets = np.random.normal(0, 0.5, (image_num, 3))
    cameras = []
    for pos, target in zip(camera_pos, targets):
        cameras.append(Camera(pos, target))

    # データ点の設定
    X = set_points1()

    # 2次元画像平面へ射影
    x_list = []
    for camera in cameras:
        x = camera.project_points(X, f, method="perspective")
        x_list.append(x)

    # ノイズの追加
    for x in x_list:
        x += 0.005 * np.random.randn(*x.shape)

    camera_poses = []
    for camera in cameras:
        camera_poses.append((camera.get_pose()))

    X_, R_ = paraperspective_self_calibration(x_list, f * np.ones(image_num))

    # 3次元点の表示
    plotter_3d = ThreeDimensionalPlotter(figsize=(10, 10))
    plotter_3d.set_lim()
    plotter_3d.plot_points(X)
    for i, camera_pose in enumerate(camera_poses, start=1):
        plotter_3d.plot_basis(camera_pose[0], camera_pose[1], label=f"Camera{i}")
    plotter_3d.show()
    plotter_3d.clear()

    # 2次元に射影したデータ点の表示
    width = 3
    height = (image_num - 1) // width + 1
    ax_list = []
    for i in range(height):
        range_width = range(image_num % width) if i == image_num // width else range(width)
        for j in range_width:
            # camera(i * j)で射影した2次元データ点のプロット
            ax_list.append(plt.subplot(height, width, width * i + j + 1))
            ax_list[width * i + j].set_title(f"Camera {width * i + j + 1}")
            ax_list[width * i + j].set_xlim(-1, 1)
            ax_list[width * i + j].set_ylim(-1, 1)
            plt.grid()
            plot_2d_points(x_list[width * i + j], ax_list[width * i + j], color="black")
    plt.show()

    # 復元したデータ点の表示
    plotter_3d = ThreeDimensionalPlotter(figsize=(10, 10))
    plotter_3d.set_lim()
    plotter_3d.plot_points(X_)
    for i, R in enumerate(R_, start=1):
        plotter_3d.plot_basis(R, -3 * R[:, 2], label=f"Camera{i}")
    plotter_3d.show()
    plotter_3d.clear()


if __name__ == "__main__":
    main()
