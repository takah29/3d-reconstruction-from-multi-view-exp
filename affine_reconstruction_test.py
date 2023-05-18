import numpy as np

from lib.affine_camera_calibration import (
    orthographic_self_calibration,
    paraperspective_self_calibration,
    symmetric_affine_self_calibration,
)
from lib.bundle_adjustment import BundleAdjuster
from lib.camera import Camera, calc_projected_points, get_camera_parames
from lib.utils import sample_hemisphere_points, set_points1
from lib.visualization import show_2d_projection_data, show_3d_scene_data


def main():
    np.random.seed(123)

    f = 1.0
    n_images = 12

    # カメラの設定
    camera_pos = sample_hemisphere_points(n_images, 5)
    targets = np.random.normal(0, 0.5, (n_images, 3))
    cameras = []
    for pos, target in zip(camera_pos, targets):
        cameras.append(Camera.create(pos, target, f))

    # カメラパラメータの取得
    K_gt, R_gt, t_gt = get_camera_parames(cameras)

    # データ点の設定
    X_gt = set_points1()

    # シーンデータの表示
    show_3d_scene_data(X_gt, R_gt, t_gt)

    # # 2次元画像平面へ射影
    x_list = calc_projected_points(X_gt, K_gt, R_gt, t_gt)

    # ノイズの追加
    for x in x_list:
        x += 0.005 * np.random.randn(*x.shape)

    X_, R_ = paraperspective_self_calibration(x_list, f * np.ones(n_images))
    t_ = -3 * R_[:, :, 2]
    K_ = np.broadcast_to(np.eye(3), R_.shape)
    # 復元したシーンデータの表示
    show_3d_scene_data(X_, R_, t_)

    # 投影データと復元後の再投影データを表示
    reproj_x_list = calc_projected_points(X_, K_, R_, t_)
    show_2d_projection_data(x_list, reproj_x_list, n_col=6)

    # バンドル調整
    print("Bundle Adjustment")
    bundle_adjuster = BundleAdjuster(x_list, X_, K_, R_, t_)
    X_, K_, R_, t_ = bundle_adjuster.optimize(delta_tol=1e-6, scale_factor=2.0)

    # バンドル調整後のシーンデータの表示
    show_3d_scene_data(X_, R_, t_)

    # 投影データとバンドル調整後の再投影データを表示
    reproj_x_list = calc_projected_points(X_, K_, R_, t_)
    show_2d_projection_data(x_list, reproj_x_list, n_col=6)


if __name__ == "__main__":
    main()
