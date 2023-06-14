import numpy as np

from lib.bundle_adjustment import BundleAdjuster
from lib.camera import Camera, calc_projected_points, get_camera_parames
from lib.perspective_camera_calibration import (
    perspective_self_calibration,
    correct_world_coordinates,
)
from lib.utils import sample_hemisphere_points, set_points1
from lib.visualization import animate, show_2d_projection_data, show_3d_scene_data


def main():
    np.random.seed(123)

    f = 1.0
    n_images = 10

    # カメラの設定
    camera_pos = sample_hemisphere_points(n_images, 5)
    targets = np.random.normal(0, 0.5, (n_images, 3))
    cameras = []
    for pos, target in zip(camera_pos, targets):
        cameras.append(Camera.create(pos, target, f=f, f0=1.0))

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

    X_, R_, t_, K_ = perspective_self_calibration(x_list, 1.0, tol=1e-2, method="dual")

    # 復元したシーンデータの表示
    show_3d_scene_data(X_, R_, t_)

    # 投影データと復元後の再投影データを表示
    reproj_x_list = calc_projected_points(X_, K_, R_, t_)
    show_2d_projection_data(x_list, reproj_x_list, n_col=6)

    # バンドル調整
    print("Bundle Adjustment")
    bundle_adjuster = BundleAdjuster(
        np.stack(x_list).transpose(1, 0, 2), X_, K_, R_, t_, axis="x-up_z-forward"
    )
    X_, K_, R_, t_ = bundle_adjuster.optimize(2.0, 1e-8, max_iter=100, is_debug=True)
    X_, R_, t_ = correct_world_coordinates(X_, R_, t_, method="predict")
    data = bundle_adjuster.get_log()

    # バンドル調整後のシーンデータの表示
    show_3d_scene_data(X_, R_, t_)

    # 投影データとバンドル調整後の再投影データを表示
    reproj_x_list = calc_projected_points(X_, K_, R_, t_)
    show_2d_projection_data(x_list, reproj_x_list, n_col=6)

    animate(data)


if __name__ == "__main__":
    main()
