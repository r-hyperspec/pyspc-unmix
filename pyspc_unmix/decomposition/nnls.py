import numpy as np
from scipy.optimize import lsq_linear
from scipy.optimize import nnls as _nnls

from .base import LinearDecomposition, OLSTransformer


def nnls(A: np.ndarray, B: np.ndarray, solver="nnls", **kwargs) -> np.ndarray:
    if B.ndim == 1:
        B = B.reshape((1, -1))
    nnls_funcs = {
        "nnls": lambda x: _nnls(A.T, x, **kwargs)[0],
        "lsq": lambda x: lsq_linear(A.T, x, bounds=(0, np.inf), **kwargs).x,
        # "cvxpy": lambda x: nnls_cvxpy(A, x, **kwargs),
    }
    return np.apply_along_axis(nnls_funcs[solver], axis=1, arr=B)


class NNLS(OLSTransformer):
    def __init__(self, A, solver="nnls", **kwargs):
        self.solver = solver
        self.kwargs = kwargs
        self.A = A

    def transform(self, X):
        return nnls(self.A, X, solver=self.solver, **self.kwargs)


class NNLSDecomposition(LinearDecomposition):
    def __init__(
        self,
        X: np.ndarray,
        ems: np.ndarray,
        solver: str = "nnls",
        names: str = "EM",
        **kwargs,
    ):
        transformer = NNLS(ems, solver=solver, **kwargs)
        scores = transformer.transform(X)

        super().__init__(
            loadings=ems,
            scores=scores,
            names=names,
            transformer=transformer,
        )
