import numpy as np
from numpy.typing import ArrayLike

from .base import LinearDecomposition, Transformer


def cart2bary(x: ArrayLike, vertices: ArrayLike) -> np.ndarray:
    """Conversion of Cartesian to Barycentric coordinates.

    Parameters
    ----------
    x : ArrayLike of shape (M, N)
        2d matrix of Cartesian coordinates to be converted to barycentric. One row corresponds to one line.
    vertices : ArrayLike of shape (N+1, N)
        Vertex points of the simples with respect to which barycentric coordinates should be computed

    Returns
    -------
    np.ndarray of shape (M,N+1)
        Barycentric coordinates of the points with respect to provided vertex points.

    Raises
    ------
    ValueError
        Dimensions of given points and vertices mismatch
    """
    x = np.array(x)
    vertices = np.array(vertices)
    m, n = x.shape

    if vertices.shape[1] != n:
        raise ValueError(
            "Vertices must have same number of columns as the point matrix"
        )

    A = vertices[:-1, :] - vertices[-1, :]
    Ainv = np.linalg.inv(A.T)
    bary_coefs = (x @ Ainv.T) - (vertices[-1, :] @ Ainv.T)
    bary_coefs = np.hstack((bary_coefs, 1 - bary_coefs.sum(axis=1).reshape((-1, 1))))

    return bary_coefs


class Bary(Transformer):
    def __init__(self, A):
        self.A = A

    def transform(self, X):
        return cart2bary(X, vertices=self.A)

    def inverse_transform(self, X):
        return X @ self.A


class BaryDecomposition(LinearDecomposition):
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
