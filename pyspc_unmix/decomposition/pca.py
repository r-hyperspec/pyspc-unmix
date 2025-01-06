import numpy as np
from sklearn.decomposition import PCA

from .base import LinearDecomposition


class PCADecomposition(LinearDecomposition):
    def __init__(self, X: np.ndarray, **kwargs):
        pca = PCA(**kwargs)
        scores = pca.fit_transform(X)

        super().__init__(
            loadings=pca.components_,
            scores=scores,
            names="PC",
            transformer=pca,
        )
