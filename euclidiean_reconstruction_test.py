import numpy as np

from lib.camera import Camera
from lib.perspective_camera_calibration import perspective_self_calibration
from lib.bundle_adjustment import BundleAdjuster
from lib.utils import sample_hemisphere_points, set_points1
from lib.visualization import ThreeDimensionalPlotter, TwoDimensionalMatrixPlotter


def main():
    np.random.seed(123)

    f = 1.0
    n_images = 12

    # カメラの設定
    camera_pos = sample_hemisphere_points(n_images, 5)
    targets = np.random.normal(0, 0.5, (n_images, 3))
    cameras = []
    for pos, target in zip(camera_pos, targets):
        cameras.append(Camera.create(pos, target, f=f, f0=1.0))

    # データ点の設定
    X = set_points1()

    # 2次元画像平面へ射影
    x_list = []
    for camera in cameras:
        x = camera.project_points(X, method="perspective")
        x_list.append(x)

    # ノイズの追加
    # for x in x_list:
    #     x += 0.005 * np.random.randn(*x.shape)

    camera_poses = []
    for camera in cameras:
        camera_poses.append((camera.get_pose()))

    X_, R_, t_, K_ = perspective_self_calibration(x_list, 1.0, tol=1e-2, method="dual")

    # シーンデータの表示
    plotter_3d = ThreeDimensionalPlotter(figsize=(10, 10))
    plotter_3d.set_lim()
    plotter_3d.plot_points(X)
    for i, camera_pose in enumerate(camera_poses, start=1):
        plotter_3d.plot_basis(camera_pose[0], camera_pose[1], label=f"Camera{i}")
    plotter_3d.show()
    plotter_3d.close()

    # 復元したシーンデータの表示
    plotter_3d = ThreeDimensionalPlotter(figsize=(10, 10))
    plotter_3d.set_lim()
    plotter_3d.plot_points(X_)
    for i, (R, t) in enumerate(zip(R_, t_), start=1):
        plotter_3d.plot_basis(R, t, label=f"Camera{i}")
    plotter_3d.show()
    plotter_3d.close()

    # 投影データと復元後の再投影データの表示
    cameras_ = []
    for R_pred, t_pred, K_pred in zip(R_, t_, K_):
        cameras_.append(Camera(R_pred, t_pred, K_pred))

    x_list_ = []
    for camera in cameras_:
        x = camera.project_points(X_, method="perspective")
        x_list_.append(x)

    n_col = 6
    n_row = (n_images - 1) // n_col + 1
    plotter_2d = TwoDimensionalMatrixPlotter(n_row, n_col, (20, 6))
    for i in range(n_row):
        range_width = range(n_images % n_col) if i == n_images // n_col else range(n_col)
        for j in range_width:
            # camera(i * j)で射影した2次元データ点のプロット
            plotter_2d.select(n_col * i + j)
            plotter_2d.set_property(f"Camera {n_col * i + j + 1}", (-0.5, 0.5), (-0.5, 0.5))

            plotter_2d.plot_points(x_list[n_col * i + j], color="green", label="Projection")
            plotter_2d.plot_points(x_list_[n_col * i + j], color="red", label="Reprojection")

    plotter_2d.show()
    plotter_2d.close()

    print("Bundle Adjustment")
    bundle_adjuster = BundleAdjuster(x_list, X_, K_, R_, t_)
    X_, K_, R_, t_ = bundle_adjuster.optimize(convergence_threshold=1e-5)

    # バンドル調整後のシーンデータの表示
    plotter_3d = ThreeDimensionalPlotter(figsize=(10, 10))
    plotter_3d.set_lim()
    plotter_3d.plot_points(X_)
    for i, (R, t) in enumerate(zip(R_, t_), start=1):
        plotter_3d.plot_basis(R, t, label=f"Camera{i}")
    plotter_3d.show()
    plotter_3d.close()

    cameras_ = []
    for R_pred, t_pred, K_pred in zip(R_, t_, K_):
        cameras_.append(Camera(R_pred, t_pred, K_pred))

    x_list_ = []
    for camera in cameras_:
        x = camera.project_points(X_, method="perspective")
        x_list_.append(x)

    n_col = 6
    n_row = (n_images - 1) // n_col + 1
    plotter_2d = TwoDimensionalMatrixPlotter(n_row, n_col, (20, 6))
    for i in range(n_row):
        range_width = range(n_images % n_col) if i == n_images // n_col else range(n_col)
        for j in range_width:
            # camera(i * j)で射影した2次元データ点のプロット
            plotter_2d.select(n_col * i + j)
            plotter_2d.set_property(f"Camera {n_col * i + j + 1}", (-0.5, 0.5), (-0.5, 0.5))

            plotter_2d.plot_points(x_list[n_col * i + j], color="green", label="Projection")
            plotter_2d.plot_points(x_list_[n_col * i + j], color="red", label="Reprojection")

    plotter_2d.show()
    plotter_2d.close()


if __name__ == "__main__":
    main()
