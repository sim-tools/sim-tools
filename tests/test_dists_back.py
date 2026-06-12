"""
Back tests for distribution classes in sim_tools.distributions module.
"""

import numpy as np
import pytest

import sim_tools.distributions as dists

SEED = 18


@pytest.mark.parametrize(
    "dist_class, params, random_seed, expected_sample",
    [
        (dists.Exponential, [5], True, [3.18711299, 2.74830843, 2.92235409]),
        (dists.Bernoulli, [0.7], True, [1, 0, 1]),
        (dists.Lognormal, [1, 0.5], True, [0.72920465, 0.52444868, 1.22966157]),
        (dists.Normal, [10, 2], True, [9.13531999, 7.73980646, 11.34768732]),
        (dists.Uniform, [5, 15], True, [8.99305769, 12.17414679, 7.80823328]),
        (dists.Triangular, [5, 10, 15], True, [9.4682534, 11.2411084, 8.74715444]),
        (dists.FixedDistribution, [42], False, [42, 42, 42]),
        (
            dists.CombinationDistribution,
            [dists.FixedDistribution(10), dists.FixedDistribution(5)],
            False,
            [15, 15, 15],
        ),
        (
            dists.GroupedContinuousEmpirical,
            [[0, 5, 10], [5, 10, 15], [1, 2, 1]],
            True,
            [6.49305769, 9.67414679, 5.30823328],
        ),
        (dists.Erlang, [10, 5], True, [7.2488537, 12.78563042, 22.58240369]),
        (dists.Weibull, [2, 10], True, [7.98387498, 7.41391722, 7.64506911]),
        (dists.Gamma, [2, 5], True, [5.8425284, 13.48364027, 29.26453341]),
        (dists.Beta, [2, 5], True, [0.15690027, 0.59810782, 0.6166126]),
        (dists.DiscreteEmpirical, [[1, 2, 3], [0.2, 0.5, 0.3]], True, [2, 3, 2]),
        (
            dists.TruncatedDistribution,
            [dists.Normal(mean=0, sigma=1, random_seed=SEED), 0],
            False,
            [0, 0, 0.67384366],
        ),
        (dists.RawDiscreteEmpirical, [[1, 2, 3, 4, 5]], True, [5, 2, 2]),
        (dists.PearsonV, [3, 10], True, [4.94772359, 2.54754784, 1.3346526]),
        (dists.PearsonVI, [2, 3, 5], True, [1.48841206, 13.79124657, 16.6039972]),
        (dists.ErlangK, [2, 5], True, [5.8425284, 13.48364027, 29.26453341]),
        (dists.Poisson, [5], True, [3, 6, 7]),
        (
            dists.Hyperexponential,
            [[0.5, 0.5], [1.0, 0.5]],
            True,
            [0.19956533, 4.0074458, 0.65655432],
        ),
    ],
)
def test_back(dist_class, params, random_seed, expected_sample):
    """
    Back tests, which checks that samples from each distribution (with random
    seeds) are consistent with those performed previously.

    Parameters
    ----------
    dist_class: Distribution
        Distribution class to test.
    params: dict
        Parameters for initialising the distribution.
    random_seed: boolean
        Whether the class accepts a random_seed parameter when initialising.
    expected_sample: np.ndarray
        List of values generated previously using the same code.
    """
    # Initialise the distribution
    if random_seed:
        dist = dist_class(*params, random_seed=SEED)
    else:
        dist = dist_class(*params)

    # Compare the sample to our expected sample
    sample = dist.sample(size=3)
    assert np.array_equal(sample.round(8), expected_sample), (
        f"Samples not equal. Generated: {sample}. Expected: {expected_sample}"
    )
