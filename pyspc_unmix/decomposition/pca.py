from copy import copy
from typing import Tuple

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

    def __getitem__(self, given: Tuple[slice, slice, slice]) -> LinearDecomposition:
        if not ((type(given) == tuple) and (len(given) == 3)):
            raise ValueError(
                "Invalid subset value. Provide 3 values in format <row, component, wl>"
            )

        i_spc, i_comp, i_wl = given
        new_loadings = self._loadings[i_comp, i_wl]
        new_scores = self._scores[i_spc, i_comp]
        new_names = self.names[i_comp]

        # Copy the transformer and update the components, explained variance, etc.
        transformer: PCA = copy(self.transformer)
        transformer.components_ = transformer.components_[i_comp, i_wl]
        transformer.explained_variance_ = transformer.explained_variance_[i_comp]
        transformer.explained_variance_ratio_ = transformer.explained_variance_ratio_[
            i_comp
        ]
        transformer.singular_values_ = transformer.singular_values_[i_comp]
        transformer.mean_ = transformer.mean_[i_wl]

        return LinearDecomposition(
            loadings=new_loadings,
            scores=new_scores,
            transformer=transformer,
            names=new_names,
        )
