import numpy as np
from numpy.typing import NDArray


class UnionFind:
    def __init__(self, n: int):
        self.parent = np.arange(n)
        self.rank = np.zeros(n)

    def find(self, x: int):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x: int, y: int):
        px = self.find(x)
        py = self.find(y)

        if px == py:
            return False

        if self.rank[px] > self.rank[py]:
            self.parent[py] = px
        else:
            self.parent[px] = py
            if self.rank[px] == self.rank[py]:
                self.rank[py] += 1

        return True


class MinimumSpanningTree:
    def __init__(self, edges: NDArray, weights: NDArray):
        """最小全域木を求めるクラス

        入力データ形式
        edges: 各ノード間の辺、shapeは(N, 2)
        costs: edgesに対応した距離（重み）、shapeは(N, )
        """

        if len(edges) != len(weights):
            raise ValueError()

        sort_ind = weights.argsort()
        self._sorted_edges = np.hstack((edges, weights[:, np.newaxis]))[sort_ind]

        self._n_nodes = np.max(edges) + 1
        self._union_find = UnionFind(self._n_nodes)

    def solve(self) -> NDArray:
        """kruskal法で最小全域木を求める"""
        res = []
        for edges in self._sorted_edges:
            if self._union_find.union(*edges[:2]):
                res.append(edges)

        result = np.vstack(res)

        return result

    def to_adjacency_matrix(self, result: NDArray) -> tuple[NDArray, NDArray]:
        """求めた最小全域木の計算結果を隣接行列と距離行列に変換する"""
        i_arr, j_arr = result[:, :2].T
        adjacency_mat = np.zeros((self._n_nodes,) * 2, dtype=np.uint8)
        adjacency_mat[i_arr, j_arr] = 1
        adjacency_mat[j_arr, i_arr] = 1

        weights = result[:, 2]
        distance_mat = np.full(adjacency_mat.shape, np.nan)
        distance_mat[i_arr, j_arr] = weights
        distance_mat[j_arr, i_arr] = weights

        return adjacency_mat, distance_mat


if __name__ == "__main__":
    edges = np.array(
        [
            (0, 1, 2),
            (0, 2, 3),
            (0, 3, 5),
            (1, 3, 7),
            (2, 4, 2),
            (3, 4, 15),
            (3, 5, 1),
            (4, 6, 11),
            (5, 6, 8),
        ]
    )
    edges, weights = np.split(edges, (2,), axis=1)
    mst = MinimumSpanningTree(edges, weights.ravel())
    min_span_tree = mst.solve()

    print(min_span_tree)
    print(*mst.to_adjacency_matrix(min_span_tree), sep="\n")
