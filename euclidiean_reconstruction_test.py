import numpy as np

from lib.bundle_adjustment import BundleAdjuster
from lib.camera import Camera
from lib.perspective_camera_calibration import perspective_self_calibration
from lib.utils import sample_hemisphere_points, set_points1
from lib.visualization import (
    ThreeDimensionalPlotter,
    TwoDimensionalMatrixPlotter,
    animate,
    show_2d_projection_data,
    show_3d_scene_data,
)


def main():
    np.random.seed(123)

    f = 1.0
    n_images = 18

    # カメラの設定
    camera_pos = sample_hemisphere_points(n_images, 5)
    targets = np.random.normal(0, 0.5, (n_images, 3))
    cameras = []
    for pos, target in zip(camera_pos, targets):
        cameras.append(Camera.create(pos, target, f=f, f0=1.0))

    # データ点の設定
    X_gt = set_points1()

    # 2次元画像平面へ射影
    x_list = []
    for camera in cameras:
        x = camera.project_points(X_gt, method="perspective")
        x_list.append(x)

    # ノイズの追加
    for x in x_list:
        x += 0.005 * np.random.randn(*x.shape)

    R_gt = []
    t_gt = []
    for camera in cameras:
        R_gt_, t_gt_ = camera.get_pose()
        R_gt.append(R_gt_)
        t_gt.append(t_gt_)
    R_gt = np.stack(R_gt)
    t_gt = np.stack(t_gt)

    X_, R_, t_, K_ = perspective_self_calibration(x_list, 1.0, tol=1e-3, method="dual")

    # シーンデータの表示
    show_3d_scene_data(X_gt, R_gt, t_gt)

    # 復元したシーンデータの表示
    show_3d_scene_data(X_, R_, t_)

    # 投影データと復元後の再投影データの表示
    cameras_ = []
    for R_pred, t_pred, K_pred in zip(R_, t_, K_):
        cameras_.append(Camera(R_pred, t_pred, K_pred))

    reproj_x_list = []
    for camera in cameras_:
        x = camera.project_points(X_, method="perspective")
        reproj_x_list.append(x)

    show_2d_projection_data(x_list, reproj_x_list, n_col=6)

    print("Bundle Adjustment")
    bundle_adjuster = BundleAdjuster(x_list, X_, K_, R_, t_)
    X_, K_, R_, t_ = bundle_adjuster.optimize(convergence_threshold=1e-3, is_debug=True)
    data = bundle_adjuster.get_log()

    # バンドル調整後のシーンデータの表示
    show_3d_scene_data(X_, R_, t_)

    cameras_ = []
    for R_pred, t_pred, K_pred in zip(R_, t_, K_):
        cameras_.append(Camera(R_pred, t_pred, K_pred))

    reproj_x_list = []
    for camera in cameras_:
        x = camera.project_points(X_, method="perspective")
        reproj_x_list.append(x)

    show_2d_projection_data(x_list, reproj_x_list, n_col=6)

    animate(data)


if __name__ == "__main__":
    main()
