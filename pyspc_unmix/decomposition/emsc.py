import numpy as np
from sklearn.preprocessing import minmax_scale

from .base import LinearDecomposition, OLSTransformer


def vandermonde(x: np.ndarray, p=3) -> np.ndarray:
    """Vandermonde matrix

    Generate a Vandermonde matrix of polynomial order `p` for a given vector `x`.

    Parameters
    ----------
    x : (n,) np.ndarray
        Input vector defining the range where the polynomial is evaluated.
    p : int, optional
        Polynomial order, by default 3. NOTE: unlike `numpy.vander`, the order
        is inclusive, i.e. p=3 will generate a matrix with 4 components.

    Returns
    -------
    (p+1, n) np.ndarray
        Vandermonde matrix of polynomial order `p`.
    """
    x = minmax_scale(x, feature_range=(-1, 1))
    return np.vander(x, p + 1).T


class EMSC(OLSTransformer):
    """EMSC transformer"""

    def __init__(self, refs: np.ndarray, p: int = 3):
        """EMSC transformer

        Parameters
        ----------
        refs : np.ndarray
            Array of reference spectra (e.g. pure components) to be used in
            EMSC fitting.
        p : int, optional
            Polynomial order to be used for background estimation, by default 3
        """
        self.refs = refs
        self.polys = vandermonde(refs.wl, p=p)
        self.p = p

    @property
    def A(self):
        return np.vstack([self.refs, self.polys])


class EMSCDecomposition(LinearDecomposition):
    """EMSC decomposition"""

    def __init__(
        self,
        X: np.ndarray,
        ems: np.ndarray,
        p: int = 3,
        names: str = "EM",
        **kwargs,
    ):
        transformer = EMSC(ems, p=p, **kwargs)
        scores = transformer.transform(X)

        super().__init__(
            loadings=ems,
            scores=scores,
            names=names,
            transformer=transformer,
        )
