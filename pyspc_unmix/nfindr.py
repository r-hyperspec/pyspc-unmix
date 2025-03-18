from typing import List, Optional, Tuple, Union
from warnings import warn

import numpy as np
from numpy.typing import ArrayLike
from scipy.optimize import lsq_linear, nnls
import scipy.linalg as la
from sklearn.base import BaseEstimator, OneToOneFeatureMixin, TransformerMixin
from sklearn.utils import check_random_state
from sklearn.utils.validation import check_is_fitted, validate_data

from .simplex import _pad_ones, _simplex_E, cart2bary, simplex_volume

__all__ = ["nfindr", "NFINDR"]


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
        Einv = la.inv(E)

    if endmembers is None:
        endmembers = range(len(indices))
    elif isinstance(endmembers, int):
        endmembers = [endmembers]

    if new_indices is None:
        new_indices = range(x.shape[0])
    elif isinstance(new_indices, int):
        new_indices = [new_indices]

    # NOTE: la.solve(E, _pad_ones(x[new_indices, :])[:, ems]) might be faster
    # however, it would use more memory. Also, Einv is not reusable in that case.
    ratios = _pad_ones(x[new_indices, :]) @ Einv.T[:, endmembers]

    return np.abs(ratios)


def _init_random(x: np.ndarray, random_state=None) -> np.ndarray:
    """Initialize NFINDR with random points"""
    m = x.shape[0]
    n = x.shape[1]
    p = n + 1
    random_state: np.random.RandomState = check_random_state(random_state)
    return random_state.choice(m, p, replace=False)


def _init_projections(x: np.ndarray, random_state=None) -> np.ndarray:
    """Initialize NFINDR with projections of data onto random vectors"""
    n = x.shape[1]
    p = n + 1
    random_state: np.random.RandomState = check_random_state(random_state)

    indices = set()
    while len(indices) < p:
        w = random_state.normal(scale=1, size=n)
        projections = np.dot(x, w)
        indices = indices.union(
            [
                np.argmax(projections),
                np.argmin(projections),
            ]
        )

    return np.array(list(indices)[:p])


def _single_nfindr_run(
    x: np.array,
    indices: List[int],
    iter_max: int = 10,
    keep_replacements: bool = False,
    tol: float = 1e-8,
) -> Tuple[List[int], List[List[int]]]:
    """Run a single NFINDR iteration"""
    p = x.shape[1] + 1
    n_iters = 0
    is_replacement = True
    indices_best = list(indices).copy()
    replacements = [indices_best.copy()]
    Einv = la.inv(_simplex_E(x, indices_best))
    while (n_iters < iter_max) and is_replacement:
        n_iters += 1
        is_replacement = False
        for j in range(p):
            estimates = _estimate_volume_change(
                x, indices_best, endmembers=j, Einv=Einv
            )
            if any(estimates > (1 + tol)):
                # Update current simplex vertices
                i, _ = np.unravel_index(np.nanargmax(estimates), estimates.shape)
                indices_best[j] = i
                Einv = la.inv(_simplex_E(x, indices_best))
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

    return indices_best, None


def nfindr(
    x: ArrayLike,
    init: Union[str, List[int]] = "projections",
    iter: str = "points",
    estimator: str = "Cramer",
    iter_max: int = 10,
    n_init: int = 1,
    keep_replacements: bool = False,
    tol: float = 1e-8,
    random_state=None,
) -> Union[List[int], Tuple[List[int], List[float], List[List[int]]]]:
    """Run N-FIND algorithm

    The implementation correspoinds to iter="points", estimator="Cramer"
    from the `unmixR` R package. The data is expected to be already
    reduced to the dimension of p-1 (where p is number of endmembers).

    Parameters
    ----------
    x : ArrayLike
        (n, p-1)-dimensional data matrix to unmix
    init : Union[str, List[int]], optional
        Initialization strategy, by default projections. If None, random initialization
        is used. Possible values are:
        - "random" - random initialization
        - "projections" -  selecting two extreme points of the projections of the data
        onto random vectors.
        - list (or an array-like) of p integers - manually selected
        initial points, can be output of another endmember extraction method, e.g. VCA
    iter : str, optional
        The iteration strategy, by default "points". Other options are not supported.
    estimator : str, optional
        Volume change estimator, by default "Cramer". Other options are not supported.
    iter_max : int, optional
        Maximum number of outer loops, by default 10
    n_init : int, optional
        Number of initializations to try. The final result will be the best output of
        all initializations. Ignored if specific initial endmember indices provided.
        For backward compatibility, by default 1.
    keep_replacements : bool, optional
        Return list of replacements as well as the list of the best indices,
        by default False. Mainly for debugging purposes.
    tol : float, optional
        Tolerance for the volume change, by default 1e-8. If the relative volume change
        is smaller than this value, the replacement is not made.
    random_state : int, RandomState instance or None, optional
        Pass an int for reproducible results across multiple function calls.
        Works the same as `random_state` in `sklearn`

    Returns
    -------
    endmember_indices: List[int]
        List of indices giving the largest volume, i.e. the found endmember points.
        The list is sorted so the output remains stable.
    volumes: List[float], if `keep_replacements` is True
        List of `n_init` simplex volumes for each initialization. The largest volume
        corresponds to the endmembers in `endmember_indices`.
    replacements: List[List[int]], if `keep_replacements` is True
        For each `n_init` initialization, the list of replacements that were made,
        i.e. the first row/element corresponds to the initialized enemembers,
        and the last is the list of the final endmembers giving the largest volume.
    """

    if iter != "points" or estimator != "Cramer":
        raise NotImplementedError(
            "The only supported combination of `iter` and `estimator` is "
            "'points' and 'Cramer'"
        )

    # Prepare data matrix
    x = np.array(x)
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
    if isinstance(init, str) and (init == "random"):
        # Random initialization
        init_indices_list = [
            _init_random(x, random_state=random_state) for _ in range(n_init)
        ]
    elif isinstance(init, str) and (init == "projections"):
        # Extremes of random projections
        init_indices_list = [
            _init_projections(x, random_state=random_state) for _ in range(n_init)
        ]
    else:
        # Manually provided initial points
        init = np.array(init)
        if len(init) != p:
            raise ValueError(
                f"Invalid number of initial points. Expected list of {p} "
                f"integer indices, but got {init}."
            )
        init = init.astype(int)
        init_indices_list = [init]

    # Run the algorithm n_init times
    nf_results = [
        _single_nfindr_run(
            x, indices, iter_max=iter_max, keep_replacements=keep_replacements, tol=tol
        )
        for indices in init_indices_list
    ]

    # Find the best result
    volumes = [simplex_volume(x[i, :], factorial=False) for i, _ in nf_results]
    best_idx = np.argmax(volumes)

    # Return the best result
    if keep_replacements:
        replacements = [r for _, r in nf_results]
        return nf_results[best_idx][0], volumes, replacements

    return nf_results[best_idx][0]


class NFINDR(OneToOneFeatureMixin, TransformerMixin, BaseEstimator):
    """NFINDR unmixing algorithm

    Finds the endmembers using NFINDR algorithm. Given the endmebers, decompose
    the data to the endmembers coefficiens using non-negative least squares (NNLS).
    The data expected to be with already reduced dimension.

    Parameters
    ----------
    n_endmembers : int, default=None
        Number of endmembers to find.

    init : Union[str, List[int]], default="projections"
        Initialization strategy. If None, random initialization is used. Possible values
        are:
        - "random" - random initialization
        - "projections" -  selecting two extreme points of the projections of the data
        onto random vectors.
        - list (or an array-like) of p integers - manually selected initial points, can
        be output of another endmember extraction method, e.g. VCA

    iter_max : int, default=10
        Maximum number of outer loops.

    n_init : int, default=1
        Number of initializations to try. The final result will be the best output of
        all initializations. Ignored if specific initial endmember indices provided.

    tol : float, default=1e-8
        Tolerance for the volume change. If the relative volume change is smaller than
        this value, the replacement is not made.

    random_state : int, RandomState instance or None, default=None
        Pass an int for reproducible results across multiple function calls.
        Works the same as random_state in `sklearn.decomposition.PCA`

    Attributes
    ----------
    endmembers_ : ndarray of shape (n_endmembers, n_endmembers-1)
        Matrix of vertex points found by NFINDR algorithm (in the reduced dimension!).

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
        init: Union[None, str, List[int]] = "projections",
        iter_max: int = 10,
        n_init: int = 1,
        # keep_replacements: bool = False,
        tol: float = 1e-8,
        random_state=None,
    ) -> None:
        self.n_endmembers = n_endmembers
        self.init = init
        self.iter_max = iter_max
        self.n_init = n_init
        # self.keep_replacements = keep_replacements
        self.tol = tol
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

        X = validate_data(self, X=X, dtype=[np.float64, np.float32], ensure_2d=True)

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

        endmember_indices = nfindr(
            X[:, : (n_endmembers - 1)],
            init=self.init,
            iter_max=self.iter_max,
            n_init=self.n_init,
            # keep_replacements=self.keep_replacements,
            keep_replacements=False,
            tol=self.tol,
            random_state=self.random_state,
        )
        self.endmember_indices_ = endmember_indices
        self.endmembers_ = X[endmember_indices, :]
        self.n_endmembers_ = n_endmembers
        # self.initial_indices_ = list(initial_indices)
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
        method: str, default="barycentric"
            Method to use for unmixing. Can be either "barycentric", "nnls", or "lsq".
            "barycentric" uses barycentric coordinates. Both "nnls" and "lsq" use
            non-negative least squares (NNLS) for unmixing. The difference is that
            "nnls" uses `scipy.optimize.nnls` and
            "lsq" uses `scipy.optimize.lsq_linear`.
            NNLS is faster but less stable than LSQ. For more details see the related
            issue: https://github.com/r-hyperspec/pyspc-unmix/issues/3

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

        X = validate_data(self, X=X, dtype=[np.float64, np.float32], reset=False)
        A = self.endmembers_.T

        if method == "barycentric":
            X_transformed = cart2bary(
                X[:, : (self.n_endmembers_ - 1)], self.endmembers_
            )
        elif method == "nnls":
            X_transformed = np.apply_along_axis(
                lambda x: nnls(A, x)[0],
                axis=1,
                arr=X[:, : (self.n_endmembers_ - 1)],
            )
        elif method == "lsq":
            X_transformed = np.apply_along_axis(
                lambda x: lsq_linear(A, x, bounds=(0, np.inf)).x,
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
