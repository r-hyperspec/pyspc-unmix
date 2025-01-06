import numpy as np
from sklearn.preprocessing import minmax_scale

from .base import LinearDecomposition, OLSTransformer


def vandermonde(x, p=3) -> np.ndarray:
    x = minmax_scale(x, feature_range=(-1, 1))
    return np.vander(x, p + 1).T


class EMSC(OLSTransformer):
    def __init__(self, refs: np.ndarray, p: int = 3):
        self.refs = refs
        self.polys = vandermonde(refs.wl, p=p)
        self.p = p

    @property
    def A(self):
        return np.vstack([self.refs, self.polys])


class EMSCDecomposition(LinearDecomposition):
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
