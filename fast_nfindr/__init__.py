__version__ = "0.1.0"

import math
from typing import List, Optional, Tuple, Union
import random
from warnings import warn
import numpy as np
from numpy.typing import ArrayLike


def _inner_simplex_points(
    vertices: ArrayLike, high: Union[float, List[float]] = 1.0, n=100
) -> np.ndarray:
    """Generate inner simplex points

    Parameters
    ----------
    vertices : ArrayLike
        List of vetex points. If a matrix, then vertex points are in rows,
        i.e. the size must be (N+1)xN
    high : float, optional
        Maximum possible coefficient for a vertex point, by default 1.0
    n : int, optional
        Number of points to generate, by default 100

    Returns
    -------
    np.ndarray
        nxN matrix of `n` points inside of the N-dimensional simplex defined
        by the vertices
    """
    vertices = np.array(vertices)
    p = len(vertices)

    coefficients = np.empty((0, p), dtype=np.float16)
    k = n - len(coefficients)
    while k > 0:
        new_coefficients = np.random.uniform(high=high, size=(3 * k, p))
        new_coefficients = new_coefficients[np.sum(new_coefficients, axis=1) <= 1, :]

        coefficients = np.vstack((coefficients, new_coefficients))

        k = n - len(coefficients)

    return coefficients[:n, :] @ vertices


def _pad_ones(x: ArrayLike) -> np.ndarray:
    """Add column of ones to a 2D matrix

    Parameters
    ----------
    x : ArrayLike
        2D matrix of size (N, M)

    Returns
    -------
    np.ndarray
        2D matrix of size (N, M+1) where the first column is a vector of 1s
        and the rest of the matrix is `x`
    """
    x = np.array(x)
    ones_column = np.ones((x.shape[0], 1), dtype=x.dtype)
    return np.hstack((ones_column, x))


def _simplex_E(x: ArrayLike, indices: Optional[List[int]] = None) -> ArrayLike:
    """Generate a simplex volume matrix

    Simple helper function for generating a simplex volume matrix E
    (i.e. volume of simplex = det(E)/p-1!) of the following structure:
    |   1   1   ... 1 |
    | e_1 e_2 ... e_p |
    Where e_i is an i-th vertex point of the simplex.

    Parameters
    ----------
    x : ArrayLike
        Matrix whose rows will be included in the simplex. This
        matrix should be reduced using using PCA or some other process
        so that it has p-1 columns before calling this function.
    indices : Optional[List[int]], optional
        Locations of the rows in the dataset to use as simplex vertecies, by default None

    Returns
    -------
    ArrayLike
        A simplex volume matrix E, a p x p matrix whose first row contains only 1s

    Raises
    ------
    ValueError
        Length (indices) does not correspond to dimensionality of data
    """
    x: np.ndarray = np.array(x)

    if indices is None:
        indices = range(x.shape[0])

    return np.transpose(_pad_ones(x[indices, :]))


def _estimate_volume_change(
    x: ArrayLike,
    indices: List[int],
    endmembers: Optional[Union[int, List[int]]] = None,
    new_indices: Optional[Union[int, List[int]]] = None,
    Einv: Optional[np.ndarray] = None,
) -> np.ndarray:
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


def simplex_volume(x: ArrayLike, factorial: bool = True) -> float:
    """Simplex volume

    Calculates a simplex volume based on determinant formula

    Parameters
    ----------
    x : ArrayLike
        A 2D matrix containing simplex vertices in rows. The number of rows expected
        to be 1 more than number of columns. As, for example, 2D simplex would be a
        trianle with three points
    factorial : bool, optional
        Whether to divide the matrix determinant by factorial, by default True

    Returns
    -------
    float
        The N-D volume of the provided simplex

    Raises
    ------
    ValueError
        The simplex matrix does not have Nx(N-1) shape
    """
    x: np.ndarray = np.array(x)

    if x.shape[0] != (x.shape[1] + 1):
        raise ValueError(
            "Unexpected array size. The simplex matrix must be of size Nx(N-1)"
        )

    volume = np.linalg.det(_pad_ones(x))

    if factorial:
        volume = volume / math.factorial(x.shape[1])

    return np.abs(volume)


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
        indices = random.sample(range(m), p)

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
