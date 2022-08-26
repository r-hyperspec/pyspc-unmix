import random
from typing import List, Optional, Tuple, Union
from warnings import warn

import numpy as np
from numpy.typing import ArrayLike

from .simplex import _pad_ones, _simplex_E


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
        N+1 indecies of the initial endmembers
    endmembers : Optional[Union[int, List[int]]], optional
        One or list of endmember indecies for replacement, by default all endmembers,
        i.e. `range(N+1)`
    new_indices : Optional[Union[int, List[int]]], optional
        One or list of point indecies for replacement, by default all points,
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
