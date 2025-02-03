import numpy as np
import pytest

from pyspc_unmix.decomposition.base import OLSTransformer


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
