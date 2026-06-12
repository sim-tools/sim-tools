"""Tests for DistributionRegistry."""

import pandas as pd
import pytest

import sim_tools.distributions as dists

REGISTRY_DISTS = [
    {
        "name": "Exponential",
        "params": {"mean": 5},
    },
    {
        "name": "Bernoulli",
        "params": {"p": 0.7},
    },
    {
        "name": "Lognormal",
        "params": {"mean": 1, "stdev": 0.5},
    },
    {
        "name": "Normal",
        "params": {"mean": 10, "sigma": 2},
    },
    {
        "name": "Uniform",
        "params": {"low": 5, "high": 15},
    },
    {
        "name": "Triangular",
        "params": {"low": 5, "mode": 10, "high": 15},
    },
    {
        "name": "FixedDistribution",
        "params": {"value": 42},
    },
    {
        "name": "CombinationDistribution",
        "params": {"dists": [dists.FixedDistribution(10), dists.FixedDistribution(5)]},
    },
    {
        "name": "GroupedContinuousEmpirical",
        "params": {
            "lower_bounds": [0, 5, 10],
            "upper_bounds": [5, 10, 15],
            "freq": [1, 2, 1],
        },
    },
    {
        "name": "Erlang",
        "params": {"mean": 10, "stdev": 5},
    },
    {
        "name": "Weibull",
        "params": {"alpha": 2, "beta": 10},
    },
    {
        "name": "Gamma",
        "params": {"alpha": 2, "beta": 5},
    },
    {
        "name": "Beta",
        "params": {"alpha1": 2, "alpha2": 5},
    },
    {
        "name": "DiscreteEmpirical",
        "params": {"values": [1, 2, 3], "freq": [0.2, 0.5, 0.3]},
    },
    {
        "name": "TruncatedDistribution",
        "params": {
            "dist_to_truncate": dists.Normal(mean=0, sigma=1),
            "lower_bound": 0,
        },
    },
    {
        "name": "RawDiscreteEmpirical",
        "params": {"values": [1, 2, 3, 4, 5]},
    },
    {
        "name": "PearsonV",
        "params": {"alpha": 3, "beta": 10},
    },
    {
        "name": "PearsonVI",
        "params": {"alpha1": 2, "alpha2": 3, "beta": 5},
    },
    {
        "name": "ErlangK",
        "params": {"k": 2, "theta": 5},
    },
    {
        "name": "Poisson",
        "params": {"rate": 5},
    },
    {
        "name": "Hyperexponential",
        "params": {"probs": [0.3, 0.7], "rates": [2.0, 1.0]},
    },
    {
        "name": "NSPPThinning",
        "params": {
            "data": pd.DataFrame(
                {
                    "t": [0.0, 60.0, 120.0],
                    "mean_iat": [10.0, 5.0, 15.0],
                }
            ),
            "random_seed1": 42,
            "random_seed2": 84,
        },
    },
]


@pytest.fixture(params=REGISTRY_DISTS, ids=[case["name"] for case in REGISTRY_DISTS])
def registry_dist(request):
    """Return a registered distribution test case."""
    return request.param


def test_registry_create(registry_dist):
    """Check that expected distributions work with create()."""
    obj = dists.DistributionRegistry.create(
        registry_dist["name"], **registry_dist["params"]
    )
    assert hasattr(obj, "sample")


def test_registry_create_batch(registry_dist):
    """Check that expected distributions work with create_batch()."""
    config = {
        "test_dist": {
            "class_name": registry_dist["name"],
            "params": registry_dist["params"],
        }
    }
    obj = dists.DistributionRegistry.create_batch(config, main_seed=123)
    assert hasattr(obj["test_dist"], "sample")


def test_registry_present(registry_dist):
    """Check that expected distributions are present in the registry."""
    assert registry_dist["name"] in dists.DistributionRegistry._registry


def test_batch_sorting():
    """Check that DistributionRegistry.create_batch() sorting works."""
    d_config = {
        "b_dist": {"class_name": "Exponential", "params": {"mean": 1}},
        "a_dist": {"class_name": "Exponential", "params": {"mean": 1}},
    }
    d_sorted = dists.DistributionRegistry.create_batch(d_config, sort=True)
    d_unsorted = dists.DistributionRegistry.create_batch(d_config, sort=False)
    assert list(d_sorted.keys()) == ["a_dist", "b_dist"]
    assert list(d_unsorted.keys()) == ["b_dist", "a_dist"]


@pytest.mark.parametrize(
    "conf, should_pass",
    [
        ({"class_name": "Exponential", "params": {"mean": 1}}, True),
        ({"class_name": "Exponential"}, False),
        ({"params": {"mean": 1}}, False),
        ({"class_name": "Exponential", "params": {"mean": 1}, "foo": 123}, False),
        ({"CLASS_NAME": "Exponential", "params": {"mean": 1}}, False),
    ],
)
def test_batch_validation(conf, should_pass):
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
