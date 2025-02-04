import numpy as np

from pyspc_unmix.decomposition.emsc import EMSCDecomposition, vandermonde
from pyspc_unmix.utils import generate_demo_mixture


def test_vandermonde():
    x = np.array([1, 2, 3, 4, 5])
    p = 3
    V = vandermonde(x, p=p)
    assert V.shape == (p + 1, len(x))
    np.testing.assert_array_almost_equal(V[0, :], [1] * len(x))


def test_emsc_decomposition():
    mixtures, true_coefs, true_ems, x = generate_demo_mixture()

    # Limit to 10 samples for speed
    mixtures = mixtures[:10, :]
    true_coefs = true_coefs[:10, :]

    # Add random polynomial baseline to mixtures
    degree = 4
    # Generate polynomial background
    np.random.seed(42)
    poly_background = (
        10 * np.random.randn(mixtures.shape[0], degree + 1) @ vandermonde(x, degree)
    )
    # Normalize to [0, 2]
    poly_background -= poly_background.min(axis=1)[:, None]
    poly_background /= 0.5 * poly_background.max(axis=1)[:, None]
    # Add polynomial background to mixtures
    poly_mixtures = mixtures + poly_background

    emsc = EMSCDecomposition(poly_mixtures, ems=true_ems, p=degree, wl=x, names="EMSC")

    assert emsc.loadings.shape == (len(true_ems) + degree + 1, poly_mixtures.shape[1])
    assert emsc.scores.shape == (poly_mixtures.shape[0], len(true_ems) + degree + 1)
    assert emsc.names.tolist() == [
        "EMSC1",
        "EMSC2",
        "EMSC3",
        "Poly0",
        "Poly1",
        "Poly2",
        "Poly3",
        "Poly4",
    ]
    np.testing.assert_array_almost_equal(emsc.scores[:, :3], true_coefs, decimal=1)
