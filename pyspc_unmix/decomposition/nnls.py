import numpy as np
from scipy.optimize import lsq_linear
from scipy.optimize import nnls as _nnls

from .base import LinearDecomposition, OLSTransformer


def nnls(A: np.ndarray, B: np.ndarray, solver="nnls", **kwargs) -> np.ndarray:
    """Generalized non-negative least squares

    This function is a wrapper around typically used `scipy.optimize.nnls` and
    `scipy.optimize.lsq_linear`.  It helpful in two ways: 1) it allows to pass
    a 2D array of B (i.e. solve for mulptiple vectors), and 2) it allows to choose
    between `nnls` and `lsq` whitin the same function, as the latter is not always
    obvious to use.

    Parameters
    ----------
    A : (m, n) np.ndarray
        Coefficient array. Same as in `scipy.optimize.nnls`.
    B : (k, n) or (n,) np.ndarray
        Right-hand side vectors. The equasion is solved for each row in B.
        If B is 1D, it is reshaped to (1, n).
    solver : str, optional
        NNLS solver to use. Currently only "nnls" and "lsq" are possible.
        By default "nnls" is used.
    **kwargs:
        Additional keyword arguments to pass to the solver function.
        See `scipy.optimize.nnls` and `scipy.optimize.lsq_linear` for details.

    Returns
    -------
    (k, m) np.ndarray
        Solution vectors for each row in B.
    """
    if B.ndim == 1:
        B = B.reshape((1, -1))
    nnls_funcs = {
        "nnls": lambda x: _nnls(A.T, x, **kwargs)[0],
        "lsq": lambda x: lsq_linear(A.T, x, bounds=(0, np.inf), **kwargs).x,
        # "cvxpy": lambda x: nnls_cvxpy(A, x, **kwargs),
    }
    return np.apply_along_axis(nnls_funcs[solver], axis=1, arr=B)


class NNLS(OLSTransformer):
    """NNNLS transformer"""

    def __init__(self, A, solver="nnls", **kwargs):
        self.solver = solver
        self.kwargs = kwargs
        self.A = A

    def transform(self, X):
        return nnls(self.A, X, solver=self.solver, **self.kwargs)


class NNLSDecomposition(LinearDecomposition):
    """NNLS decomposition"""

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
