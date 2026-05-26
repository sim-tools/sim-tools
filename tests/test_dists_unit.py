"""
Unit tests for distribution classes in sim_tools.distributions module.

This module contains test functions that verify the inheritance hierarchy,
sample type correctness, and specific properties of various probability
distribution implementations.
"""

import numpy as np
import pandas as pd
import pytest

import sim_tools.distributions as dists


SEED_1 = 42


@pytest.mark.parametrize("dist_class", [
    dists.Exponential, dists.Bernoulli, dists.Lognormal, dists.Normal,
    dists.Uniform, dists.Triangular, dists.FixedDistribution,
    dists.CombinationDistribution, dists.GroupedContinuousEmpirical,
    dists.Erlang, dists.Weibull, dists.Gamma, dists.Beta,
    dists.DiscreteEmpirical, dists.TruncatedDistribution,
    dists.RawDiscreteEmpirical, dists.PearsonV, dists.PearsonVI, dists.ErlangK,
    dists.Poisson, dists.Hyperexponential
    ]
)
def test_inheritance(dist_class):
    """
    Test that all distribution inherit from the base Distribution class.

    Parameters
    ----------
    dist_class: Distribution
        Distribution class to test.
    """
    assert issubclass(dist_class, dists.Distribution)


@pytest.mark.parametrize(
    "dist_class, params, random_seed, expected_mean, expected_type",
    [
        (dists.Exponential, [5], True, 5, float),
        (dists.Bernoulli, [0.7], True, 0.7, int),
        (dists.Lognormal, [1, 0.5], True, 1, float),
        (dists.Normal, [10, 2], True, 10, float),
        (dists.Uniform, [5, 15], True, 10, float),
        (dists.Triangular, [5, 10, 15], True, 10, float),
        (dists.FixedDistribution, [42], False, 42, int),
        (dists.CombinationDistribution,
         [dists.FixedDistribution(10), dists.FixedDistribution(5)],
         False, 15, float),
        (dists.GroupedContinuousEmpirical,
         [[0, 5, 10], [5, 10, 15], [1, 2, 1]],
         True, 7.5, float),
        (dists.Erlang, [10, 5], True, 10, float),
        (dists.Weibull, [2, 10], True, 8.862, float),
        (dists.Gamma, [2, 5], True, 10, float),
        (dists.Beta, [2, 5], True, 2/7, float),
        (dists.DiscreteEmpirical, [[1, 2, 3], [0.2, 0.5, 0.3]],
         True, 2.1, int),
        (dists.TruncatedDistribution,
         [dists.Normal(mean=0, sigma=1), 0],
         False, 0.4, float | int),
        (dists.RawDiscreteEmpirical, [[1, 2, 3, 4, 5]], True, 3, int),
        (dists.PearsonV, [3, 10], True, 5, float),
        (dists.PearsonVI, [2, 3, 5], True, 5, float),
        (dists.ErlangK, [2, 5], True, 10, float),
        (dists.Poisson, [5], True, 5, int),
        (dists.Hyperexponential, [[0.3, 0.7], [2.0, 1.0]], True, 0.85, float)
    ]
)
def test_sample(dist_class, params, random_seed, expected_mean, expected_type):
    """
    Test properties of generated samples.

    This test will check that the:
    1. Data type is correct.
    2. Number of samples matches the requested size.
    3. Sample mean matches the expected mean.
    4. Random seed (if relevant) is working.

    Parameters
    ----------
    dist_class: Distribution
        Distribution class to test.
    params: dict
        Parameters for initialising the distribution.
    random_seed: boolean
        Whether the class accepts a random_seed parameter when initialising.
    expected_mean: float
        Expected mean of the distribution.
    expected_type:
        Expected type of the sample (float for continuous distributions,
        int for discrete).
    """
    # Initialise the distribution
    dist = dist_class(*params)

    # Check that type of a single sample is as expected
    x = dist.sample()
    assert isinstance(x, expected_type), (
        f"Expected sample() to return a {expected_type} - instead: {type(x)}"
    )

    # Check that type of multiple samples is as expected
    samples = dist.sample(size=10000)
    assert isinstance(samples, np.ndarray)

    # Check that the sample size matches the requested size
    assert len(samples) == 10000

    # Check that the mean of generated samples is close to the expected mean
    assert np.isclose(np.mean(samples), expected_mean, rtol=0.1), (
        f"Expected mean {expected_mean}, actual mean {np.mean(samples)}"
    )

    if random_seed:
        # Check that the same seed returns the same sample
        sample1 = dist_class(*params, random_seed=5).sample(size=5)
        sample2 = dist_class(*params, random_seed=5).sample(size=5)
        assert np.array_equal(sample1, sample2), (
            "Samples with the same random seeds should be equal."
        )
        # Check that different seeds return different samples
        sample3 = dist_class(*params, random_seed=89).sample(size=5)
        assert not np.array_equal(sample1, sample3), (
            "Samples with different random seeds should not be equal."
        )


def test_lognormal_moments():
    """
    Test the calculation of normal distribution parameters (mu, sigma) from
    lognormal parameters.

    This test verifies that:
    1. The normal_moments_from_lognormal method correctly converts lognormal
    parameters (mean, variance) to normal distribution parameters (mu, sigma).
    2. The calculated values match the expected mathematical formulas.
    """
    # Define lognormal parameters
    mean, stdev = 2.0, 0.5

    # Initialise distribution and get calculated parameters
    dist = dists.Lognormal(mean=mean, stdev=stdev, random_seed=42)
    calculated_mu, calculated_sigma = (
        dist.normal_moments_from_lognormal(mean, stdev**2))

    # Verify calculated parameters match expected mathematical formulas
    # Formula for mu: ln(mean²/√(stdev² + mean²))
    assert np.isclose(calculated_mu,
                      np.log(mean**2 / np.sqrt(stdev**2 + mean**2)),
                      rtol=1e-5)
    # Formula for sigma: √ln(1 + stdev²/mean²)
    assert np.isclose(calculated_sigma,
                      np.sqrt(np.log(1 + stdev**2 / mean**2)),
                      rtol=1e-5)


def test_fixed_value():
    """
    Check that the FixedDistribution sample method returns the same value as
    input.
    """
    d = dists.FixedDistribution(5.0)
    assert d.sample() == 5.0


def test_discrete_probabilities():
    """
    Test correct calculation of probabilities for Discrete distribution.

    This test verifies that:
    1. The Discrete class correctly normalises frequency values to
    probabilities.
    2. The sum of probabilities equals 1.
    3. The relative proportions match the input frequencies.
    """
    # Define discrete distribution parameters
    values = [1, 2, 3]
    freq = [10, 20, 30]

    # Initialise distribution
    dist = dists.DiscreteEmpirical(values=values, freq=freq, random_seed=42)

    # Calculate expected probabilities by normalising frequencies
    expected_probs = np.array(freq) / np.sum(freq)

    # Verify calculated probabilities match expected values
    assert np.allclose(dist.probabilities, expected_probs, rtol=1e-5)

    # Verify probabilities sum to 1
    assert np.isclose(np.sum(dist.probabilities), 1.0, rtol=1e-10)


def test_discrete_value_error():
    """
    Test if Discrete raises ValueError for mismatched inputs.

    This test verifies that the Discrete class correctly validates that
    the values and frequencies arrays have the same length.
    """
    # Attempt to initialise with mismatched array lengths
    with pytest.raises(ValueError):
        dists.DiscreteEmpirical(values=[1, 2], freq=[0.5], random_seed=42)


def test_discrete_uneven_probabilities():
    """
    Test behavior of Discrete distribution with highly uneven probabilities.

    This test verifies that when one probability is much larger than another,
    the sampling correctly reflects this imbalance by rarely selecting the
    low-probability value.
    """
    # Create distribution with extremely uneven probabilities
    dist = dists.DiscreteEmpirical(
        values=[1, 2], freq=[1, 1e-10], random_seed=42)

    # Generate a large sample
    samples = dist.sample(size=10000)

    # Verify that the low-probability value (2) appears very rarely
    # With p ≈ 1e-10, we expect virtually no occurrences of value 2
    assert np.sum(samples == 2) < 5


@pytest.mark.parametrize(
    "n, lower_bound",
    [
        (1, 10.0),
        (10, 10.0),
        (100, 10.0),
        (10_000_000, 10.0),
        (10_000_000, 0.0),
        (10_000_000, 0.01),
    ],
)
def test_truncated_min(n, lower_bound):
    """
    Check that samples from the TruncatedDistribution do not fall below the
    specified lower bound.
    """
    d1 = dists.Normal(10, 1, random_seed=SEED_1)
    d2 = dists.TruncatedDistribution(d1, lower_bound=lower_bound)
    assert min(d2.sample(size=n)) >= lower_bound


def test_registry_batch_sorting():
    """Check that DistributionRegistry.create_batch() sorting works."""
    d_config = {
        "b_dist": {"class_name": "Exponential", "params": {"mean": 1}},
        "a_dist": {"class_name": "Exponential", "params": {"mean": 1}}
    }
    d_sorted = dists.DistributionRegistry.create_batch(d_config, sort=True)
    d_unsorted = dists.DistributionRegistry.create_batch(d_config, sort=False)
    assert list(d_sorted.keys()) == ["a_dist", "b_dist"]
    assert list(d_unsorted.keys()) == ["b_dist", "a_dist"]


@pytest.mark.parametrize("conf, should_pass", [
    ({"class_name": "Exponential", "params": {"mean": 1}}, True),
    ({"class_name": "Exponential"}, False),
    ({"params": {"mean": 1}}, False),
    ({"class_name": "Exponential", "params": {"mean": 1}, "foo": 123}, False),
    ({"CLASS_NAME": "Exponential", "params": {"mean": 1}}, False),
])
def test_registry_batch_validation(conf, should_pass):
    """
    Check that DistributionRegistry.create_batch() warns for unsuitable
    distribution configurations.
    """
    seed = 123
    if should_pass:
        obj = dists.DistributionRegistry._validate_and_create(conf, seed)
        assert hasattr(obj, "sample")
    else:
        with pytest.raises(ValueError):
            dists.DistributionRegistry._validate_and_create(conf, seed)


def test_registry_nsppthinning():
    """Check that Registry includes time_dependent.NSPPThinning."""
    # NSPPThinning is in a different module - confirm it is registered and can
    # still be created via the registry.
    assert "NSPPThinning" in dists.DistributionRegistry._registry, (
        "NSPPThinning should be registered in DistributionRegistry"
    )
