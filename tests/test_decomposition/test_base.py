import numpy as np
import pytest

from pyspc_unmix.decomposition.base import OLSTransformer, LinearDecomposition
from pyspc_unmix.utils import generate_demo_mixture


@pytest.fixture
def sample_data():
    # Create sample data and transformation matrix
    S = 100 * np.array(
        [
            [0.5, 0.3, 0.2],
            [0.2, 0.5, 0.3],
        ]
    )
    n_samples = 5
    n_components, n_features = S.shape
    C = np.arange(n_samples * n_components).reshape(n_samples, n_components)
    X = C @ S  # + np.random.randn(n_samples, n_features) * 0.1
    return X, C, S


def test_ols_basic_transform(sample_data):
    X, C, S = sample_data
    transformer = OLSTransformer(S)

    # Test transform
    transformed = transformer.transform(X)
    assert np.allclose(transformed, C)

    # Test inverse transform
    reconstructed = transformer.inverse_transform(transformed)
    assert np.allclose(reconstructed, X)

    # Check shapes
    assert transformed.shape == C.shape
    assert reconstructed.shape == X.shape


def test_decomposition_and_slicing():
    X, C, S, wl = generate_demo_mixture()

    # Limit to 10 samples for speed
    X = X[:10, :]
    C = C[:10, :]

    # Test decomposition
    dc = LinearDecomposition(loadings=S, scores=C)

    assert dc.loadings.shape == S.shape
    assert dc.scores.shape == C.shape
    assert dc.names.tolist() == ["Comp1", "Comp2", "Comp3"]
    assert isinstance(dc.transformer, OLSTransformer)

    # Test slicing
    dc_slice = dc[:, 1:, :]
    np.testing.assert_equal(dc_slice.loadings, S[1:, :])
    np.testing.assert_equal(dc_slice.scores, C[:, 1:])

    dc_slice = dc[:, 1:, 20:30]
    np.testing.assert_equal(dc_slice.loadings, S[1:, 20:30])
    np.testing.assert_equal(dc_slice.scores, C[:, 1:])

    dc_slice = dc[[0, 5], 1:, 20:30]
    np.testing.assert_equal(dc_slice.loadings, S[1:, 20:30])
    np.testing.assert_equal(dc_slice.scores, C[[0, 5], 1:])

    # Single value slicing
    dc_slice = dc[4, 1:, :]
    np.testing.assert_equal(dc_slice.loadings, S[1:, :])
    np.testing.assert_equal(dc_slice.scores, C[4, 1:])
    assert dc_slice.D.shape == (len(wl),)
    assert dc_slice.scores.shape == (2,)
