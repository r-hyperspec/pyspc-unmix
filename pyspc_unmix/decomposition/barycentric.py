import numpy as np

from .base import LinearDecomposition, Transformer
from ..simplex import cart2bary


class Bary(Transformer):
    """Barycentric transformation

    Given a set of vertices, this transformer converts cartesian coordinates
    to barycentric coordinates and vice versa.
    """

    def __init__(self, A):
        self.A = A

    def transform(self, X):
        return cart2bary(X, vertices=self.A)

    def inverse_transform(self, X):
        return X @ self.A


class BaryDecomposition(LinearDecomposition):
    """Barycentric decomposition"""

    def __init__(
        self,
        X: np.ndarray,
        ems: np.ndarray,
        names: str = "EM",
    ):
        transformer = Bary(ems)
        scores = transformer.transform(X)

        super().__init__(
            loadings=ems,
            scores=scores,
            transformer=transformer,
            names=names,
        )
