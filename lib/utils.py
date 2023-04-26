import numpy as np
from numpy.typing import NDArray


def unit_vec(x: NDArray) -> NDArray:
    """単位ベクトルを求める"""
    return x / np.linalg.norm(x)


def sample_normal_dist(scale: float, n: int):
    return np.random.normal(0, scale, (n, 3))


def add_noise(X, scale: float) -> NDArray:
    return X + np.random.normal(0, scale, X.shape)


def sample_hemisphere_points(num, r):
    points = []
    for _ in range(num):
        theta = np.random.uniform(0, np.pi / 2)
        phi = np.random.uniform(0, 2 * np.pi)

        x = r * np.cos(theta)
        y = r * np.sin(theta) * np.cos(phi)
        z = r * np.sin(theta) * np.sin(phi)

        points.append((x, y, z))

    return points


def set_points1():
    points = []
    for x in np.linspace(-1, 1, 10):
        for theta in np.linspace(np.pi / 2, 3 * np.pi / 2, 20):
            r = 1 / (x + 2)
            y, z = r * np.cos(theta), r * np.sin(theta)
            points.append((x, y, z))

    return np.vstack(points)
