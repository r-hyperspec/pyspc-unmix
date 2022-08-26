import numpy as np
import pytest

from pyspc_unmix.nfindr import _estimate_volume_change, nfindr
from pyspc_unmix.simplex import _inner_simplex_points, simplex_volume


@pytest.fixture
def points_2d():
    return np.array(
        [
            # initial points
            [0, 0],
            [3, 0],
            [0, 4],
            # new points
            [2, 2],
            [5, 3],
            [4, 10],
        ]
    )


@pytest.fixture
def replacement_volumes(points_2d):
    """Calculate precise volume changes

    Vij is the volume of a simples if endmember `j` is replaced by new point `i`
    """
    m = points_2d.shape[0]
    p = points_2d.shape[1] + 1
    volumes = np.empty((m, p), dtype=float)
    for i in range(m):
        for j in range(p):
            inx = list(range(p))
            inx[j] = i
            volumes[i, j] = simplex_volume(points_2d[inx, :])
    return volumes


@pytest.mark.parametrize(
    "endmembers,new_indices",
    [(None, None), (None, 1), (1, None), (1, 1), ([0, 1], [2, 3])],
)
def test_volume_change_estimator(
    points_2d, replacement_volumes, endmembers, new_indices
):
    m = points_2d.shape[0]
    p = points_2d.shape[1] + 1
    indices = range(3)
    V = replacement_volumes[0, 0]

    estimates = _estimate_volume_change(
        points_2d, indices, endmembers=endmembers, new_indices=new_indices
    )

    if endmembers is None:
        endmembers = range(p)
    elif isinstance(endmembers, int):
        endmembers = [endmembers]

    if new_indices is None:
        new_indices = range(m)
    elif isinstance(new_indices, int):
        new_indices = [new_indices]
    ref_volumes = replacement_volumes[np.ix_(new_indices, endmembers)]

    np.testing.assert_array_almost_equal(V * estimates, ref_volumes)


@pytest.fixture
def simplex_points():
    vertices = [[-5, 0], [0, 4], [10, 0]]
    initial_points = [[0, 0], [-1, 0], [0, 1]]
    data = np.vstack([initial_points, vertices, _inner_simplex_points(vertices)])
    return data


def test_nfindr(simplex_points):
    best_indices = nfindr(simplex_points)
    assert best_indices == [3, 4, 5]

    best_indices = nfindr(simplex_points, indices=range(3))
    assert best_indices == [3, 4, 5]

    best_indices, replacements = nfindr(
        simplex_points, indices=range(3), keep_replacements=True
    )
    assert best_indices == [3, 4, 5]
    assert replacements == [
        [0, 1, 2],
        [5, 1, 2],
        [5, 4, 2],
        [5, 4, 3],
    ]
