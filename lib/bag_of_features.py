from pathlib import Path
from typing import Self

import cv2
import numpy as np
from numpy.typing import NDArray
from sklearn.cluster import KMeans


class ImageDataset:
    def __init__(self, image_dir_path: str | Path):
        self._image_dir_path = Path(image_dir_path).resolve()
        self._image_path_list = sorted(list(self._image_dir_path.glob("*")))
        self._n_images = len(self._image_path_list)

        self._i = 0

    def __getitem__(self, item: int | slice) -> NDArray | list[NDArray]:
        if isinstance(item, slice):
            result = []
            for pth in self._image_path_list[item.start : item.stop : item.step]:
                result.append(cv2.imread(str(pth)))
            return result
        else:
            return cv2.imread(str(self._image_path_list[item]))

    def __len__(self):
        return self._n_images

    def __iter__(self):
        return self

    def __next__(self):
        if self._i == self._n_images:
            raise StopIteration()

        img = self[self._i]
        self._i += 1

        return img


class BagOfFeatures:
    def __init__(self, feature_extractor, clustering_algorithm):
        self._feature_extractor = feature_extractor
        self._clustering_algorithm = clustering_algorithm

    def fit(self, image_dataset: ImageDataset) -> Self:
        res = []
        for img in image_dataset:
            _, des = self._feature_extractor.detectAndCompute(img, None)
            res.append(des)

        X = np.vstack(res)
        self._clustering_algorithm.fit(X)

        return self

    def transform(self, image_dataset: ImageDataset) -> NDArray:
        res = []
        for img in image_dataset:
            _, des = self._feature_extractor.detectAndCompute(img, None)
            cluster_num_arr = self._clustering_algorithm.predict(des)
            v = np.bincount(cluster_num_arr, minlength=self._clustering_algorithm.n_clusters) / len(
                cluster_num_arr
            )
            res.append(v)

        return np.vstack(res)

    @staticmethod
    def create(feature_ext_method, clustring_method, n_features=50):
        if feature_ext_method == "SIFT":
            feature_extractor = cv2.SIFT_create()
        else:
            raise NotImplementedError

        if clustring_method == "KMeans":
            clustering_algorithm = KMeans(n_features, n_init="auto")
        else:
            raise NotImplementedError

        return BagOfFeatures(feature_extractor, clustering_algorithm)


if __name__ == "__main__":
    image_dataset = list(ImageDataset("./images"))
    bow = BagOfFeatures.create("SIFT", "KMeans")
    bow.fit(image_dataset)
    result = bow.transform(image_dataset[:1])

    print(result)
