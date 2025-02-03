import warnings
from abc import ABC, abstractmethod
from copy import copy, deepcopy
from typing import List, Optional, Tuple, Union

import numpy as np


class Transformer(ABC):
    def fit(self, X: np.ndarray, y=None):
        warnings.warn("Fitting is ignored. No training is required.")

    def fit_transform(self, X: np.ndarray, y=None) -> np.ndarray:
        warnings.warn("Fitting is ignored. No training is required.")
        return self.transform(X)

    @abstractmethod
    def transform(self, X: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        pass


class OLSTransformer(Transformer):
    """Ordinary Least Squares (OLS) Transformer.

    Transforms data using ordinary least squares regression against
    a fixed transformation matrix A.

    Parameters
    ----------
    A : np.ndarray
        Fixed transformation matrix of shape (n_features, n_components)

    Notes
    -----
    The transformation is performed by solving the equation X = SA
    where:
    - X is the input data matrix of shape (n_samples, n_features)
    - S is the transformed data matrix of shape (n_samples, n_components)
    - A is the fixed transformation matrix

    The least squares solution is computed using numpy.linalg.lstsq with rcond=None,
    which uses the SVD-based algorithm to find the solution that minimizes ||X - SA||_2.
    """

    def __init__(self, A: np.ndarray):
        self.A = A

    def transform(self, X: np.ndarray) -> np.ndarray:
        return np.linalg.lstsq(self.A.T, X.T, rcond=None)[0].T

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        return X @ self.A


class LinearDecomposition:
    """Base class for linear decomposition

    This class is a base class for storing lenear decompositoins using a transformer,
    i.e. $D = SC$, using MCR-ALS notation where
    * D = original spectral data,
    * S = spectra (loadings in PCA),
    * C = concentrations (scores in PCA).
    This is a useful class for storing results of common linear decomposition methods,
    such as OLS, PCA, NNLS, EMSC, etc.
    The default implementation uses OLS transformer.

    Parameters
    ----------
    loadings: np.ndarray
        Loadings matrix of shape (n_components, n_features)
    scores: np.ndarray
        Scores matrix of shape (n_samples, n_components)
    transformer : Transformer, optional
        Transformer to use for transformation. If None, an OLSTransformer is used.
    names : str or list of str, optional
        Names of the components. If a string is provided, it is used for all components,
        e.g. "EM" for endmember, "PC" for principal component, etc.

    """

    def __init__(
        self,
        loadings: np.ndarray,
        scores: np.ndarray,
        transformer: Optional[Transformer] = None,
        names: Union[str, List] = "Comp",
    ):
        # Get the transformer
        if transformer is None:
            self.transformer = OLSTransformer(loadings)
        else:
            self.transformer = transformer

        # Get names
        if isinstance(names, str):
            n_comp = loadings.shape[0]
            n_digits = len(str(n_comp))
            names = [f"{names}{str(i+1).zfill(n_digits)}" for i in range(n_comp)]

        self._loadings = loadings
        self._scores = scores
        self._names = np.array(names).flatten()

    @classmethod
    def from_transformer(
        cls, X, transformer: Transformer, loadins_from: str, names="Comp"
    ):
        scores = transformer.fit_transform(X)
        loadings = getattr(transformer, loadins_from)
        return cls(
            loadings=loadings,
            scores=scores,
            transformer=transformer,
            names=names,
        )

    def __copy__(self):
        """Create a shallow copy of the LinearDecomposition instance."""
        return LinearDecomposition(
            loadings=self._loadings.copy(),
            scores=self._scores.copy(),
            transformer=copy(self.transformer),
            names=self._names.copy(),
        )

    def __deepcopy__(self, memo):
        """Create a deep copy of the LinearDecomposition instance."""
        return LinearDecomposition(
            loadings=deepcopy(self._loadings, memo),
            scores=deepcopy(self._scores, memo),
            transformer=deepcopy(self.transformer, memo),
            names=deepcopy(self._names, memo),
        )

    # ----------------------------------------------------------------------
    # Properties for a quick access
    @property
    def names(self) -> np.ndarray:
        return self._names

    @names.setter
    def names(self, new_names: List[str]):
        new_names = np.array(new_names).reshape((-1,))
        if len(new_names) != self.ncomp:
            raise ValueError(
                f"Number of new names ({len(new_names)}) should be equal "
                f"to the number of components ({self.ncomp})"
            )

        self._names = new_names

    @property
    def ncomp(self):
        return self._loadings.shape[0]

    @property
    def scores(self) -> np.ndarray:
        """Concentrations (scores)"""
        return self._scores

    @property
    def C(self) -> np.ndarray:
        """Concentrations (scores)"""
        return self._scores

    @property
    def loadings(self) -> np.ndarray:
        """Loadings (spectra)"""
        return self._loadings

    @property
    def S(self) -> np.ndarray:
        """Loadings (spectra)"""
        return self._loadings

    @property
    def D(self) -> np.ndarray:
        """Reconstructed data (D)"""
        return self.transformer.inverse_transform(self._scores)

    @property
    def nwl(self) -> int:
        """Number of wavelength points"""
        return len(self._loadings.shape[1])

    @property
    def nspc(self) -> int:
        """Number of spectra in the object"""
        return self._scores.shape[0]

    @property
    def shape(self) -> tuple:
        return (self.nspc, self.ncomp, self.nwl)

    # ----------------------------------------------------------------------
    # Accessing data
    def __getitem__(self, given: Tuple[slice, slice, slice]) -> "LinearDecomposition":
        """Get a subset of the decomposition data.

        Parameters
        ----------
        given : tuple
            A 3-tuple containing (row indices, component indices, wavelength indices)
            to subset the decomposition data.

        Returns
        -------
        LinearDecomposition
            A new LinearDecomposition object containing the subset of data.

        Examples
        --------
        >>> import numpy as np
        >>> from pyspc_unmix.decomposition import LinearDecomposition
        >>> # Create sample data
        >>> loadings = np.random.rand(3, 100)  # 3 components, 100 wavelengths
        >>> scores = np.random.rand(50, 3)     # 50 samples, 3 components
        >>> decomp = LinearDecomposition(loadings=loadings, scores=scores)
        >>>
        >>> # Get subset of first 10 samples, first 2 components, all wavelengths
        >>> subset = decomp[:10, :2, :]
        >>> print(subset.shape)
        (10, 2, 100)
        >>> print(subset.names)
        ['Comp1', 'Comp2']
        >>> print(subset.D)
        """
        if not ((type(given) == tuple) and (len(given) == 3)):
            raise ValueError(
                "Invalid subset value. Provide 3 values in format <row, component, wl>"
            )

        new_loadings = self._loadings[given[1], given[2]]
        new_scores = self._scores[given[0], given[1]]

        # TODO: Handle different types of Decomposition. E.g. PCA, etc.
        return LinearDecomposition(
            loadings=new_loadings,
            scores=new_scores,
            names=self.names[given[1]],
        )

    # ----------------------------------------------------------------------
    # spc2scores and scores2spc
    def transform(self, X: np.ndarray) -> np.ndarray:
        return self.transformer.transform(X)

    def inverse_transform(self, scores: np.ndarray) -> np.ndarray:
        return self.transformer.inverse_transform(scores)

    # ----------------------------------------------------------------------
    # Plotting
    def pairplot(self, components=None, diag=False, **kwargs):
        import matplotlib.pyplot as plt

        if components is None:
            components = 5

        if isinstance(components, int):
            components = range(min(self.ncomp, components))

        offset = 0 if diag else 1
        scores = self.scores[:, components]
        fig, axs = plt.subplots(
            ncols=len(components) - offset,
            nrows=len(components) - offset,
            layout="tight",
            sharex=True,
            sharey=True,
        )

        for j in range(len(components) - 1):
            for i in range(j + 1, len(components)):
                ax = axs[i - offset, j]
                ax.scatter(scores[:, components[j]], scores[:, components[i]], **kwargs)
                ax.set_xlabel(self.names[components[j]])
                ax.set_ylabel(self.names[components[i]])

        return fig, axs
