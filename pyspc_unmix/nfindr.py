from typing import List, Optional, Tuple, Union
from warnings import warn

import numpy as np
from numpy.typing import ArrayLike
from scipy.optimize import nnls
from sklearn.base import BaseEstimator, TransformerMixin, OneToOneFeatureMixin
from sklearn.utils import check_random_state
from sklearn.utils.validation import check_is_fitted

from .simplex import _pad_ones, _simplex_E, cart2bary, simplex_volume


def _estimate_volume_change(
    x: ArrayLike,
    indices: List[int],
    endmembers: Optional[Union[int, List[int]]] = None,
    new_indices: Optional[Union[int, List[int]]] = None,
    Einv: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Estimate volume change using Cramer's rule

    Parameters
    ----------
    x : ArrayLike
        Matrix of M points in N-dimensional space
    indices : List[int]
        N+1 indices of the initial endmembers
    endmembers : Optional[Union[int, List[int]]], optional
        One or list of endmember indices for replacement, by default all endmembers,
        i.e. `range(N+1)`
    new_indices : Optional[Union[int, List[int]]], optional
        One or list of point indices for replacement, by default all points,
        i.e. `range(M)`
    Einv : Optional[np.ndarray], optional
        Pre-calculated inversed E matrix for faster calculation, by default None

    Returns
    -------
    np.ndarray
        LxK matrix (V), where L is the length of `new_indices` and K is the length of
        `endmembers`, where `Vij` estimates how would the simplex volume chage if the
        j-th endmember would be replaced by i-th point. The calculated value is the
        new volume divided by old (initial) volume.
    """
    x = np.array(x)
    if Einv is None:
        E = _simplex_E(x, indices)
        Einv = np.linalg.inv(E)

    if endmembers is None:
        endmembers = range(len(indices))
    elif isinstance(endmembers, int):
        endmembers = [endmembers]

    if new_indices is None:
        new_indices = range(x.shape[0])
    elif isinstance(new_indices, int):
        new_indices = [new_indices]

    ratios = _pad_ones(x[new_indices, :]) @ Einv.T[:, endmembers]

    return np.abs(ratios)


def nfindr(
    x: ArrayLike,
    indices: Optional[List[int]] = None,
    iter_max: int = 10,
    keep_replacements: bool = False,
) -> Union[List[int], Tuple[List[int], List[List[int]]]]:
    """Run N-FIND algorithm

    The implementation correspoinds to iter="points", estimator="Cramer"
    from the `unmixR` R package.

    Parameters
    ----------
    x : ArrayLike
        N-dimensional data matrix
    indices : Optional[List[int]], optional
        List of the initial points indices, by default generated randomly
    iter_max : int, optional
        Maximum number of outer loops, by default 10
    keep_replacements : bool, optional
        Return list of replacements as well as the list of the best indices,
        by default False

    Returns
    -------
    endmember_indices: List[int]
        List of indices giving the largest volume, i.e. the found endmember points.
        The list is sorted so the output would be more stable.
    replacements: List[List[int]], if `keep_replacements` is True
        List of sets of candidate points that were iterated over. I.e. the first
        row/element is the list of initial points, the last is the list of the final
        points giving the largest volume.
    """

    # Prepare data matrix
    x = np.array(x)
    m = x.shape[0]
    n = x.shape[1]

    # # Validate number of components
    # if not (isinstance(p, int) and (p > 2)):
    #     raise ValueError(
    #         f"Invalid number of endmembers for search. "
    #         "Please provide an integer number greater than 2."
    #     )

    # if n != p - 1:
    #     raise ValueError(
    #         "Mismatching number of endmembers and data dimension. "
    #         "The data dimension (number of columns) must be equal to p-1."
    #     )
    p = n + 1

    # Get initial indices
    if indices is None:
        indices = np.random.choice(range(m), p, replace=False)

    n_iters = 0
    is_replacement = True
    indices_best = list(indices).copy()
    replacements = [indices_best.copy()]
    Einv = np.linalg.inv(_simplex_E(x, indices_best))
    while (n_iters < iter_max) and is_replacement:
        n_iters += 1
        is_replacement = False
        for j in range(p):
            estimates = _estimate_volume_change(
                x, indices_best, endmembers=j, Einv=Einv
            )
            if any(estimates > (1 + 1.5e-8)):
                # Update current simplex vertices
                i, _ = np.unravel_index(np.nanargmax(estimates), estimates.shape)
                indices_best[j] = i
                Einv = np.linalg.inv(_simplex_E(x, indices_best))
                # Mark that a replacement took place
                is_replacement = True
                # For debugging
                if keep_replacements:
                    replacements.append(indices_best.copy())

    if j == iter_max:
        warn(
            "The maximum number of iterations was reached. "
            "The iterator was interrupted."
        )

    # Sort the values to have same output if the endmebers are the same
    indices_best.sort()

    if keep_replacements:
        return indices_best, replacements

    return indices_best


class NFINDR(OneToOneFeatureMixin, TransformerMixin, BaseEstimator):
    """NFINDR unmixing algorithm

    Finds the endmembers using NFINDR algorithm. Given the endmebers, decompose
    the data to the endmembers coefficiens using non-negative least squares (NNLS).
    The data expected to be with already reduced dimension.

    Parameters
    ----------
    n_endmembers : int, default=None
        Number of endmembers to find.

    initial_indices : List[int], default=None
        List of row indices to be used as initial points for NFINDR

    random_state : int, RandomState instance or None, default=None
        Pass an int for reproducible results across multiple function calls.
        Works the same as random_state in `sklearn.decomposition.PCA`

    Attributes
    ----------
    endmembers_ : ndarray of shape (n_endmembers, n_endmembers-1)
        Matrix of vertex points found by NFINDR algorithm

    initial_indices_ : List[int] of len (n_endmembers,)
        List of initial points indices.

    endmember_indices_ : List[int] of len (n_endmembers,)
        List of final endmember points indices.

    n_samples_ : int
        Number of samples in the training data.

    n_endmembers_ : int
        Number of endmembers estimated during the training. I.e. either number of
        columns in the training data + 1 or explicitly provided `n_endmembers`

    volume_ : float
        The volume of the simplex fomed by `endmembers_` vertex points

    Examples
    --------
    >>> import numpy as np
    >>> from pyspc_unmix import NFINDR
    >>> X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
    >>> nf = NFINDR()
    >>> nf.fit(X)
    NFINDR()
    >>> print(nf.endmembers_)
    [[-1. -1.]
     [-2. -1.]
     [ 3.  2.]]
    >>> print(nf.transform(X))
    [[1.00000000e+00 0.00000000e+00 7.85046229e-17]
     [0.00000000e+00 1.00000000e+00 0.00000000e+00]
     [1.00000000e+00 1.00000000e+00 0.00000000e+00]
     [0.00000000e+00 1.00000000e+00 1.00000000e+00]
     [1.00000000e+00 0.00000000e+00 1.00000000e+00]
     [0.00000000e+00 0.00000000e+00 1.00000000e+00]]
    """

    def __init__(
        self,
        n_endmembers=None,
        initial_indices=None,
        random_state=None,
    ) -> None:
        self.n_endmembers = n_endmembers
        self.initial_indices = initial_indices
        self.random_state = random_state

    def fit(self, X, y=None):
        """Fit the model with X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data, where `n_samples` is the number of samples
            and `n_features` is the number of features.

        y : Ignored
            Ignored.

        Returns
        -------
        self : object
            Returns the instance itself.
        """

        X = self._validate_data(X, dtype=[np.float64, np.float32], ensure_2d=True)

        n_samples, n_features = X.shape

        n_endmembers = self.n_endmembers or (n_features + 1)

        if n_endmembers > n_features + 1:
            raise ValueError(
                "Dimension of data is too high. Please, reduce it (e.g. by PCA) "
                "or use `fit_transform` to directly reduce the dimensionality and "
                "apply NFINDR"
            )
        elif n_endmembers < n_features + 1:
            raise ValueError(
                "Dimension of the data is too low. "
                "Please consider reducing the number of components"
            )

        if self.initial_indices is None:
            random_state: np.random.RandomState = check_random_state(self.random_state)
            initial_indices = random_state.choice(
                range(n_samples), n_endmembers, replace=False
            )
        else:
            initial_indices = self.initial_indices

        endmember_indices = nfindr(X[:, : (n_endmembers - 1)], initial_indices)
        self.endmember_indices_ = endmember_indices
        self.endmembers_ = X[endmember_indices, :]
        self.n_endmembers_ = n_endmembers
        self.initial_indices_ = list(initial_indices)
        self.n_samples_ = n_samples
        self.volume_ = simplex_volume(self.endmembers_)

        return self

    def transform(self, X, method="barycentric"):
        """Transform X to endmembers coefficients.

        X is converted to coefficients of previously found endmembers

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            New data, where `n_samples` is the number of samples
            and `n_features` is the number of features.

        Returns
        -------
        X_new : array-like of shape (n_samples, n_endmembers)
            Decomposition of X to the endmember coefficients, where `n_samples`
            is the number of samples and `n_endmembers` is the number of the endmembers

        Notes
        -----
        The same pre-treatment (e.g. PCA) must be applied to the X as it was for
        the data used for fitting.
        """
        check_is_fitted(self)

        X = self._validate_data(X, dtype=[np.float64, np.float32], reset=False)

        if method == "barycentric":
            X_transformed = cart2bary(
                X[:, : (self.n_endmembers_ - 1)], self.endmembers_
            )
        elif method == "nnls":
            X_transformed = np.apply_along_axis(
                lambda x: nnls(self.endmembers_.T, x)[0],
                axis=1,
                arr=X[:, : (self.n_endmembers_ - 1)],
            )
        else:
            raise ValueError(
                f"Unexpected method '{method}'. Must be either 'barycentric' or 'nnls'."
            )

        return X_transformed

    def inverse_transform(self, X):
        """Transform data back to its original space.

        In other words, return an input `X_original` whose transform would be X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_endmembers)
            New data, where `n_samples` is the number of samples
            and `n_endmembers` is the number of endmembers.

        Returns
        -------
        X_original array-like of shape (n_samples, n_features)
            Original data, where `n_samples` is the number of samples
            and `n_features` is the number of features.
        """
        return np.array(X) @ self.endmembers_

    def fit_transform(self, X, y=None):
        """Fit the model with X and apply unmixing on X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data, where `n_samples` is the number of samples
            and `n_features` is the number of features.

        y : Ignored
            Ignored.

        Returns
        -------
        X_new : ndarray of shape (n_samples, n_endmembers)
            Transformed values.
        """
        self.fit(X)
        return self.transform(X)

    @property
    def _n_features_out(self):
        """Number of transformed output features."""
        return self.endmembers_.shape[0]
