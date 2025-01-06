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


def test_ols_with_centering(sample_data):
    X, C, S = sample_data
    center = np.array([1.0, 2.0, 3.0])
    transformer = OLSTransformer(S, center=center)

    # Test transform
    transformed = transformer.transform(X + center)
    assert np.allclose(transformed, C)
    # Test inverse transform
    reconstructed = transformer.inverse_transform(transformed)
    assert np.allclose(reconstructed - center, X)


def test_ols_with_scaling(sample_data):
    X, C, S = sample_data
    scale = np.array([2.0, 1.0, 3.0])
    transformer = OLSTransformer(S, scale=scale)

    # Test transform
    transformed = transformer.transform(X * scale)
    assert np.allclose(transformed, C)
    # Test inverse transform
    reconstructed = transformer.inverse_transform(transformed)
    assert np.allclose(reconstructed / scale, X)


def test_ols_with_center_and_scale(sample_data):
    X, C, S = sample_data
    center = np.array([1.0, 2.0, 3.0])
    scale = np.array([2.0, 1.0, 3.0])
    transformer = OLSTransformer(S, center=center, scale=scale)

    # Test transform
    transformed = transformer.transform(X * scale + center)
    assert np.allclose(transformed, C)
    # Test inverse transform
    reconstructed = transformer.inverse_transform(transformed)
    assert np.allclose(reconstructed - center, X * scale)


def test_invalid_scale():
    S = np.array([[1.0, 0.0], [0.0, 1.0]])
    scale = np.array([1.0, 0.0])  # Contains zero

    with pytest.raises(ValueError, match="Scale cannot contain zero values"):
        OLSTransformer(S, scale=scale)


def test_shape_broadcasting():
    S = np.array([[1.0, 0.0], [0.0, 1.0]])
    center = np.array([1.0, 2.0])
    scale = np.array([2.0, 1.0])

    transformer = OLSTransformer(S, center=center, scale=scale)

    assert transformer.center.shape == (1, 2)
    assert transformer.scale.shape == (1, 2)
