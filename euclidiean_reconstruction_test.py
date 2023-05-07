import numpy as np

from lib.camera import Camera
from lib.perspective_camera_calibration import perspective_self_calibration
from lib.utils import sample_hemisphere_points, set_points1
from lib.visualization import ThreeDimensionalPlotter, TwoDimensionalMatrixPlotter


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
    # for x in x_list:
    #     x += 0.005 * np.random.randn(*x.shape)

    camera_poses = []
    for camera in cameras:
        camera_poses.append((camera.get_pose()))

    X_ = perspective_self_calibration(x_list, 1.0, method="primary")

    # 3次元点の表示
    plotter_3d = ThreeDimensionalPlotter(figsize=(10, 10))
    plotter_3d.set_lim()
    plotter_3d.plot_points(X)
    for i, camera_pose in enumerate(camera_poses, start=1):
        plotter_3d.plot_basis(camera_pose[0], camera_pose[1], label=f"Camera{i}")
    plotter_3d.show()
    plotter_3d.close()

    # 2次元に射影したデータ点の表示
    n_row = 3
    n_col = (image_num - 1) // n_row + 1
    plotter_2d = TwoDimensionalMatrixPlotter(n_row, n_col)
    for i in range(n_col):
        range_width = range(image_num % n_row) if i == image_num // n_row else range(n_row)
        for j in range_width:
            # camera(i * j)で射影した2次元データ点のプロット
            plotter_2d.select(n_row * i + j)
            plotter_2d.set_property(f"Camera {n_row * i + j + 1}", (-1, 1), (-1, 1))
            plotter_2d.plot_points(x_list[n_row * i + j], color="black")
    plotter_2d.show()
    plotter_2d.close()

    # 復元したデータ点の表示
    plotter_3d = ThreeDimensionalPlotter(figsize=(10, 10))
    plotter_3d.set_lim()
    plotter_3d.plot_points(X_)
    # for i, R in enumerate(R_, start=1):
    #     plotter_3d.plot_basis(R, -3 * R[:, 2], label=f"Camera{i}")
    plotter_3d.show()
    plotter_3d.close()


if __name__ == "__main__":
    main()
