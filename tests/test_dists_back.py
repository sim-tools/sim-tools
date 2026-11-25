"""
Back tests for distribution classes.
"""

import numpy as np
import pandas as pd
import pytest

import sim_tools.distributions as dists
from sim_tools.distributions import DistributionRegistry
from sim_tools.time_dependent import NSPPThinning

# ============================================================================
# Test cases
# ============================================================================

SEED = 18

NSPP_DATA = pd.DataFrame(
    {
        "t": [0, 1440, 2880, 4320, 5760, 6720, 7200],
        "mean_iat": [5.0, 6.0, 4.5, 7.0, 5.5, 6.5, 5.0],
    }
)

TEST_CASES = [
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
]

REGISTRY_CASES = [
    ("Exponential", {"mean": 5.0}),
    ("Bernoulli", {"p": 0.7}),
    ("Lognormal", {"mean": 1.0, "stdev": 0.5}),
    ("Normal", {"mean": 10.0, "sigma": 2.0}),
    ("Uniform", {"low": 5.0, "high": 15.0}),
    ("Triangular", {"low": 5.0, "mode": 10.0, "high": 15.0}),
    ("FixedDistribution", {"value": 42}),
    (
        "GroupedContinuousEmpirical",
        {"lower_bounds": [0, 5, 10], "upper_bounds": [5, 10, 15], "freq": [1, 2, 1]},
    ),
    ("Erlang", {"mean": 10.0, "stdev": 5}),
    ("Weibull", {"alpha": 2.0, "beta": 10.0}),
    ("Gamma", {"alpha": 2.0, "beta": 5.0}),
    ("Beta", {"alpha1": 2.0, "alpha2": 5.0}),
    ("DiscreteEmpirical", {"values": [1, 2, 3], "freq": [0.2, 0.5, 0.3]}),
    ("RawDiscreteEmpirical", {"values": [1, 2, 3, 4, 5]}),
    ("PearsonV", {"alpha": 3.0, "beta": 10.0}),
    ("PearsonVI", {"alpha1": 2.0, "alpha2": 3.0, "beta": 5.0}),
    ("ErlangK", {"k": 2, "theta": 5}),
    ("Poisson", {"rate": 5.0}),
    ("Hyperexponential", {"probs": [0.5, 0.5], "rates": [1.0, 0.5]}),
]


# ============================================================================
# Directly calling distribution
# ============================================================================


@pytest.mark.parametrize("dist_class, params, random_seed, expected_sample", TEST_CASES)
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
    sample = dist.sample(size=3).round(8)
    assert np.array_equal(sample, expected_sample), (
        f"Samples not equal. Generated: {sample}. Expected: {expected_sample}"
    )


def _sample_three_iats(dist: NSPPThinning) -> list[float]:
    """Helper that mimics model usage: sequential calls with advancing time."""
    t = 0.0
    samples = []
    for _ in range(3):
        iat = dist.sample(simulation_time=t)
        samples.append(round(iat, 8))
        t += iat
    return samples


def test_nspp_thinning_back():
    """Back test for NSPPThinning."""
    dist = NSPPThinning(
        data=NSPP_DATA,
        random_seed1=SEED,
        random_seed2=SEED + 1,
    )
    samples = _sample_three_iats(dist)
    expected = [2.86840169, 5.10359626, 0.89804398]
    assert np.array_equal(samples, expected), (
        f"NSPPThinning direct back-test failed. "
        f"Generated: {samples}. Expected: {expected}"
    )


# ============================================================================
# Calling distribution via DistributionRegistry
# ============================================================================


@pytest.mark.parametrize("class_name, params", REGISTRY_CASES)
def test_registry(class_name, params):
    """
    Check that DistributionRegistry.create_batch produces identical samples
    when called twice with the same main_seed.
    """
    config = {"dist": {"class_name": class_name, "params": params}}

    batch1 = DistributionRegistry.create_batch(
        config, main_seed=SEED, sort=True, preserve_structure=True
    )
    batch2 = DistributionRegistry.create_batch(
        config, main_seed=SEED, sort=True, preserve_structure=True
    )

    sample1 = batch1["dist"].sample(size=5)
    sample2 = batch2["dist"].sample(size=5)

    assert np.array_equal(sample1, sample2), (
        f"{class_name}: registry samples differ between runs with same seed. "
        f"sample1={sample1}, sample2={sample2}"
    )


def test_nspp_thinning_back_via_registry():
    """Regression test for NSPPThinning+DistributionRegistry"""
    config = {
        "arrivals": {
            "class_name": "NSPPThinning",
            "params": {"data": NSPP_DATA},
        }
    }

    batch1 = DistributionRegistry.create_batch(
        config, main_seed=SEED, sort=True, preserve_structure=True
    )
    batch2 = DistributionRegistry.create_batch(
        config, main_seed=SEED, sort=True, preserve_structure=True
    )

    samples1 = _sample_three_iats(batch1["arrivals"])
    samples2 = _sample_three_iats(batch2["arrivals"])

    assert np.array_equal(samples1, samples2), (
        f"NSPPThinning via registry not reproducible. "
        f"samples1={samples1}, samples2={samples2}"
    )
