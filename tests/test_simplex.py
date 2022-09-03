import numpy as np
import pytest

from pyspc_unmix.simplex import _simplex_E, cart2bary, simplex_volume


def test_simplex_E_default():
    x = [[1, 2], [3, 4], [5, 6]]
    E = _simplex_E(x)
    np.testing.assert_array_equal(E, np.array([[1, 1, 1], [1, 3, 5], [2, 4, 6]]))
    np.testing.assert_array_equal(_simplex_E(x), _simplex_E(np.array(x)))


def test_simplex_E_indices():
    x = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    np.testing.assert_array_equal(_simplex_E(x[1:, :]), _simplex_E(x, [1, 2, 3]))


def test_simplex_volume():
    triangle_2d = np.array(
        [
            [0, 0],
            [1, 0],
            [0, 1],
        ]
    )
    shifted_triangle_2d = triangle_2d + np.array([1, 3])
    area_2d = 0.5

    assert simplex_volume(triangle_2d) == pytest.approx(area_2d)
    assert simplex_volume(triangle_2d, factorial=True) == pytest.approx(area_2d)
    # Without factorial
    assert simplex_volume(triangle_2d, factorial=False) == pytest.approx(2 * area_2d)
    # Moving the triangle should not change the area
    assert simplex_volume(shifted_triangle_2d) == pytest.approx(area_2d)


def test_cart2bary_simplex():
    vertices = np.array(
        [
            [0, 0],
            [1, 0],
            [0, 1],
        ]
    )
    new_points = np.array(
        [
            [0.5, 0.5],
            [0.1, 0.2],
            [0.7, 0.1],
            [1.0, 1.0],
        ]
    )
    bary_coordinates = cart2bary(new_points, vertices)
    np.testing.assert_almost_equal(bary_coordinates.sum(axis=1), np.ones((4,)))
    np.testing.assert_almost_equal(
        bary_coordinates, np.hstack(([[0.0], [0.7], [0.2], [-1.0]], new_points))
    )

    # Shifting all points should not change the barycentric coordinates
    shift = np.array([10, 20])
    bary_coordinates_with_shift = cart2bary(new_points + shift, vertices + shift)
    np.testing.assert_almost_equal(bary_coordinates_with_shift, bary_coordinates)
