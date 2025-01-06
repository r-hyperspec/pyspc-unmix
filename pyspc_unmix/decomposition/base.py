import warnings
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple, Union

import numpy as np
from numpy.typing import ArrayLike


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

    Transforms data using ordinary least squares regression against a fixed transformation matrix A.
    Optionally applies centering and scaling before transformation.

    Parameters
    ----------
    A : np.ndarray
        Fixed transformation matrix of shape (n_features, n_components)
    center : Optional[ArrayLike], default=None
        Optional centering vector to subtract from input data
    scale : Optional[ArrayLike], default=None
        Optional scaling vector to divide input data by

    Notes
    -----
    The transformation is performed by solving the equation X = SA
    (or `(X - center) / scale = SA` if `center` and `scale` are provided)
    where:
    - X is the input data matrix of shape (n_samples, n_features)
    - S is the transformed data matrix of shape (n_samples, n_components)
    - A is the fixed transformation matrix

    The least squares solution is computed using numpy.linalg.lstsq with rcond=None,
    which uses the SVD-based algorithm to find the solution that minimizes ||X - SA||_2.
    """

    def __init__(
        self,
        A: np.ndarray,
        center: Optional[ArrayLike] = None,
        scale: Optional[ArrayLike] = None,
    ):
        self.A = A
        if center is not None:
            # Ensure center is a 1D array that will broadcast correctly
            self.center = np.array(center).reshape(1, -1)
        else:
            self.center = None

        if scale is not None:
            # Ensure scale is a 1D array that will broadcast correctly
            self.scale = np.array(scale).reshape(1, -1)
            # Prevent division by zero
            if np.any(self.scale == 0):
                raise ValueError("Scale cannot contain zero values")
        else:
            self.scale = None

    def transform(self, X: np.ndarray) -> np.ndarray:
        X_transformed = X.copy()
        if self.center is not None:
            X_transformed -= self.center
        if self.scale is not None:
            X_transformed = X_transformed / self.scale
        return np.linalg.lstsq(self.A.T, X_transformed.T, rcond=None)[0].T

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        result = X @ self.A
        if self.scale is not None:
            result = result * self.scale
        if self.center is not None:
            result = result + self.center
        return result


class LinearDecomposition:
    """Base class for linear decomposition

    This class is a base class for storing lenear decompositoins,
    i.e. $D = SC$ if using MCR-ALS notation
    (D = original spectral data, S = spectra (loadings in PCA), C = concentrations (scores in PCA)),
    original spectral data $D$ can be linearly preprocessed (i.e. centered and scaled)
    using a transformer.
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
            self._transformer = OLSTransformer(loadings)
        else:
            self._transformer = transformer

        # Get names
        if isinstance(names, str):
            names = [f"{names}{i+1}" for i in range(loadings.shape[0])]

        self._loadings = loadings
        self._scores = scores
        self._names = np.array(names).reshape((-1,))

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
                f"Number of new names ({len(new_names)}) should be equal to the number of components ({self.ncomp})"
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
        return self._transformer.inverse_transform(self._scores)

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
        return self._transformer.transform(X)

    def inverse_transform(self, scores: np.ndarray) -> np.ndarray:
        return self._transformer.inverse_transform(scores)

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
