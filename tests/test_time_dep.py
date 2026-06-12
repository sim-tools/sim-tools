"""
Basic tests for time-dependent module
"""

import pytest

import sim_tools.time_dependent as td
from sim_tools.datasets import load_banks_et_al_nspp

SEED_1 = 42
SEED_2 = 101


@pytest.mark.parametrize(
    "simulation_time",
    [
        (0.0),
        (10.0),
        (55.0),
        (10.5),
        (2_000.00),
        (10_000.00),
    ],
)
def test_sample_from_nsppthinning(simulation_time):
    """
    Check that the NSPPThinning sample method returns a float when called with
    different simulation times.
    """
    data = load_banks_et_al_nspp()
    d = td.NSPPThinning(data, SEED_1, SEED_2)
    assert isinstance(d.sample(simulation_time), float)
