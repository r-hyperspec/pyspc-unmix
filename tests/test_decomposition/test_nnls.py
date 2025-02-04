import numpy as np
import pytest
from pyspc_unmix.decomposition.nnls import nnls, NNLS, NNLSDecomposition


def test_nnls_basic():
    # Simple test case with known solution
    A = np.array([[1, 0], [0, 1]])
    b = np.array([2, 1])
    expected = np.array([[2, 1]])

    result = nnls(A, b)
    np.testing.assert_array_almost_equal(result, expected)


def test_nnls_multiple_vectors():
    A = np.array([[1, 0], [1, 1]])
    B = np.array([[2, 1], [3.5, 3]])

    result_nnls = nnls(A, B, solver="nnls")
    result_lsq = nnls(A, B, solver="lsq")

    np.testing.assert_array_almost_equal(result_nnls, [[1, 1], [0.5, 3]])
    np.testing.assert_array_almost_equal(result_lsq, [[1, 1], [0.5, 3]])


def test_nnls_invalid_solver():
    A = np.array([[1, 0], [0, 1]])
    B = np.array([[1, 1]])

    with pytest.raises(KeyError):
        nnls(A, B, solver="invalid_solver")


def test_nnls_transformer():
    A = np.array([[1, 0], [1, 1]])
    X = np.array([[2, 1]])

    transformer = NNLS(A, solver="nnls")
    result = transformer.transform(X)

    # Test if transform gives same result as direct nnls
    expected = nnls(A, X, solver="nnls")
    np.testing.assert_array_almost_equal(result, expected)

    np.testing.assert_array_almost_equal(transformer.inverse_transform(expected), X)


def test_nnls_decomposition():
    # Test the full decomposition class
    X = np.array([[2, 1]])
    ems = np.array([[1, 0], [1, 1]])

    decomp = NNLSDecomposition(X, ems)

    # Test if loadings are correct
    np.testing.assert_array_equal(decomp.loadings, ems)

    # Test if scores are correct
    expected_scores = nnls(ems, X, solver="nnls")
    np.testing.assert_array_almost_equal(decomp.scores, expected_scores)

    # Test if names are generated correctly
    np.testing.assert_array_equal(decomp.names, ["EM1", "EM2"])
