from typing import Tuple, Union

import numpy as np
from numpy.typing import ArrayLike
from scipy import stats


def generate_mixtures(
    ems: Union[int, ArrayLike],
    features: Union[int, ArrayLike],
    n_mixtures: int,
    nonnegative: bool = True,
    sum_to_one: bool = True,
    noise=None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate synthetic mixtures"""

    if isinstance(features, int):
        features = np.arange(features)
    else:
        features = np.array(features)

    if isinstance(ems, int):
        n_ems = ems
        # Generate n_ems random gaussian signals in range given by `features`
        centers = np.random.choice(features, size=n_ems, replace=False)
        dx = np.diff(features).mean()
        sigmas = np.random.choice(np.linspace(0.1 * dx, 2 * dx, 10), size=n_ems)
        ems = np.array(
            [
                stats.norm.pdf(features, loc=center, scale=sigma)
                for center, sigma in zip(centers, sigmas)
            ]
        )
    else:
        ems = np.array(ems)
        n_ems = ems.shape[0]

    # Generate random mixing coefficients
    true_coeffs = np.random.rand(n_mixtures, n_ems) * 10
    if nonnegative:
        true_coeffs = np.abs(true_coeffs)
    if sum_to_one:
        true_coeffs /= true_coeffs.sum(axis=1, keepdims=True)

    # Generate mixtures
    mixtures = true_coeffs @ ems
    if noise is not None:
        mixtures += np.random.normal(scale=noise, size=mixtures.shape)

    return mixtures, true_coeffs, ems, features


def generate_demo_mixture() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate a demo mixture for testing and reproducible examples"""
    x = np.arange(600, 900.1, 2.5)
    ems = 100 * np.array(
        [
            stats.norm.pdf(x, loc=650, scale=10),
            stats.norm.pdf(x, loc=750, scale=20),
            stats.norm.pdf(x, loc=850, scale=30),
        ]
    )
    # Set random seed for reproducibility
    np.random.seed(42)
    mixtures, true_coefs, true_ems, x = generate_mixtures(ems, x, 100, noise=0.02)

    # Insert 'almost' pure mixtures, where one component is dominant (95%)
    # Make sure that they come in the same order as the true endmembers
    # This is to test the NFINDR algorithm
    for i, i_replacement in enumerate([15, 41, 95]):
        true_coefs[i_replacement, :] = 0.05 / (len(ems) - 1)
        true_coefs[i_replacement, i] = 0.95
        mixtures[i_replacement, :] = (
            (true_coefs[[i_replacement], :] @ ems)
            + np.random.normal(scale=0.02, size=(1, len(x)))
        ).reshape((-1,))

    return mixtures, true_coefs, true_ems, x


# fig, axs = plt.subplots(2,1); axs[0].plot(wl, X.T); axs[1].plot(wl, S.T);  plt.show()
