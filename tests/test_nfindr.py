import random

import numpy as np
import pytest

from pyspc_unmix.nfindr import _estimate_volume_change, nfindr, NFINDR
from pyspc_unmix.simplex import _inner_simplex_points, cart2bary, simplex_volume


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


def test_nfindr_class_fit(simplex_points):
    vertices = [[-5, 0], [0, 4], [10, 0]]
    X = _inner_simplex_points(vertices, high=0.4)

    nf = NFINDR()
    nf.fit(np.vstack((vertices, X)))
    assert nf.endmember_indecies_ == list(range(3))
    np.testing.assert_array_equal(nf.endmembers_, vertices)

    initial_indices = random.sample(range(len(X)), 3)
    nf = NFINDR(initial_indecies=initial_indices)
    nf.fit(X)
    assert nf.endmember_indecies_ == nfindr(X, indices=initial_indices)
    np.testing.assert_array_equal(nf.endmembers_, X[nf.endmember_indecies_, :])

    # The result is different if random state is not provided
    nf1 = NFINDR()
    nf2 = NFINDR()
    nf1.fit(X)
    nf2.fit(X)
    assert nf1.initial_indecies_ != nf2.initial_indecies_

    # The result is stable if random state is provided
    nf1 = NFINDR(random_state=0)
    nf2 = NFINDR(random_state=0)
    nf1.fit(X)
    nf2.fit(X)
    assert nf1.initial_indecies_ == nf2.initial_indecies_
    assert nf1.endmember_indecies_ == nf2.endmember_indecies_


def test_nfindr_class_transform(simplex_points):
    nf = NFINDR()
    X = simplex_points

    # check the shape of fit.transform
    X_r = nf.fit(X).transform(X)
    assert X_r.shape[0] == len(X)
    assert X_r.shape[1] == 3

    # check the equivalence of fit.transform and fit_transform
    X_r2 = nf.fit_transform(X)
    np.testing.assert_array_almost_equal(X_r, X_r2)

    X_r = nf.transform(X)
    np.testing.assert_array_almost_equal(X_r, X_r2)


def test_nfindr_class_transform_barycentric():
    vertices = np.array([[0, 0], [1, 0], [0, 1]])
    points = np.array([[0.5, 0.5], [0.1, 0.2], [0.7, 0.1], [1.0, 1.0]])

    nf = NFINDR()
    nf.fit(vertices)
    assert nf.endmember_indecies_ == [0, 1, 2]
    np.testing.assert_array_equal(nf.endmembers_, vertices)

    coords = nf.transform(points)
    np.testing.assert_array_almost_equal(coords, cart2bary(points, vertices))

    # Barycentric coordinates is default
    coords2 = nf.transform(points, method="barycentric")
    np.testing.assert_array_equal(coords2, coords2)

    # Add a shift
    shift = np.array([10, 20])

    nf.fit(vertices + shift)
    assert nf.endmember_indecies_ == [0, 1, 2]
    np.testing.assert_array_equal(nf.endmembers_, vertices + shift)

    coords = nf.transform(points + shift)
    np.testing.assert_array_almost_equal(coords, cart2bary(points, vertices))


def test_nfindr_class_transform_nnls():
    vertices = np.array([[0, 0], [1, 0], [0, 1]])
    points = np.array([[0.5, 0.5], [0.1, 0.2], [0.7, 0.1], [1.0, 1.0]])
    coords = NFINDR().fit(vertices).transform(points, method="nnls")
    np.testing.assert_array_almost_equal(
        coords, np.hstack((np.zeros((len(points), 1)), points))
    )

    vertices = np.array([[1, 1], [3, 1], [1, 3]])
    points = np.array([[2, 2], [3, 3]])
    coords = NFINDR().fit(vertices).transform(points, method="nnls")
    np.testing.assert_array_almost_equal(coords, [[0, 0.5, 0.5], [0, 0.75, 0.75]])
