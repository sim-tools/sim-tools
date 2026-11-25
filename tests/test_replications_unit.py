"""Replication unit tests

Unit testing for code related to selecting the number of replications.

Credit: Some of these tests are adapted from-
    Heather, A. Monks, T. (2025). Python DES RAP Template. GitHub.
    https://github.com/pythonhealthdatascience/rap_template_python_des.
"""

import warnings
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
import scipy.stats as st

from sim_tools.output_analysis import (
    OnlineStatistics,
    ReplicationsAlgorithm,
    ReplicationTabulizer,
    confidence_interval_method,
)

# -----------------------------------------------------------------------------
# ReplicationsAlgorithm
# -----------------------------------------------------------------------------


@pytest.mark.parametrize(
    "look_ahead, n, exp", [(100, 100, 100), (100, 101, 101), (0, 500, 0)]
)
def test_klimit(look_ahead, n, exp):
    """
    Check that the _klimit() calculations are as expected.

    Parameters
    ----------
    look_ahead: int
        Minimum additional replications to look ahead to assess stability of
        precision.
    n: int
        Number of replications that would already be completed.
    exp: int
        Expected number of replications for _klimit() to return.
    """
    # Calculate additional replications that would be required
    calc = ReplicationsAlgorithm(look_ahead=100, initial_replications=100)._klimit()
    # Check that this meets our expected value
    assert calc == 100, (
        f"With look_ahead {look_ahead} and n={n}, the additional "
        + f"replications required should be {exp} but _klimit() returned {calc}."
    )


@pytest.mark.parametrize(
    "arg, value",
    [
        ("initial_replications", -1),
        ("initial_replications", 0.5),
        ("look_ahead", -1),
        ("look_ahead", 0.5),
        ("half_width_precision", 0),
    ],
)
def test_algorithm_invalid(arg, value):
    """
    Ensure that ReplicationsAlgorithm responds appropriately to invalid inputs.

    Parameters
    ----------
    arg: string
        Name of input for ReplicationsAlgorithm.
    value: float
        Value of input to ReplicationsAlgorithm.
    """
    with pytest.raises(ValueError):
        ReplicationsAlgorithm(**{arg: value})


def test_algorithm_invalid_budget():
    """
    Ensure that ReplicationsAlgorithm responds appropriately when
    replication_budget is less than initial_replications.
    """
    with pytest.raises(ValueError):
        ReplicationsAlgorithm(initial_replications=10, replication_budget=9)


@pytest.mark.parametrize(
    "lst, exp, look_ahead",
    [
        ([None, None, 0.8, 0.4, 0.3], 4, 0),  # Normal case
        ([0.4, 0.3, 0.2, 0.1], 1, 0),  # No None values
        ([0.8, 0.9, 0.8, 0.7], None, 0),  # No values below threshold
        ([None, None, None, None], None, 0),  # No values
        ([], None, 0),  # Empty list
        ([None, None, 0.8, 0.8, 0.3, 0.3, 0.3], None, 3),  # Not full lookahead
        ([None, None, 0.8, 0.8, 0.3, 0.3, 0.3, 0.3], 5, 3),  # Meets lookahead
    ],
)
def test_find_position(lst, exp, look_ahead):
    """
    Test the find_position() method from ReplicationsAlgorithm.

    Parameters
    ----------
    lst : list
        List of values to input to find_position().
    exp : float
        Expected result from find_position().
    look_ahead : int
        Number of extra positions to check that they also fall under the
        threshold.
    """
    # Set threshold to 0.5, with provided look_ahead
    alg = ReplicationsAlgorithm(half_width_precision=0.5, look_ahead=look_ahead)
    # Get result from algorithm and compare to expected
    result = alg.find_position(lst)
    assert result == exp, (
        f"Ran find_position on: {lst} (threshold 0.5, look-ahead "
        f"{look_ahead}). Expected {exp}, but got {result}."
    )


# -----------------------------------------------------------------------------
# OnlineStatistics
# -----------------------------------------------------------------------------


def test_onlinestat_data():
    """
    Check that OnlineStatistics will fail if an invalid data type is provided.
    """
    with pytest.raises(ValueError):
        OnlineStatistics(data=pd.Series([9, 2, 3]))


def test_onlinestat_computations():
    """
    Feed three values into OnlineStatistics and verify mean, standard
    deviation, confidence intervals, and deviation calculations.
    """
    values = [10, 20, 30]

    # Provide three values
    stats = OnlineStatistics(data=np.array(values), alpha=0.05, observer=None)

    # Expected values
    expected_mean = np.mean(values)
    expected_std = np.std(values, ddof=1)
    expected_lci, expected_uci = st.t.interval(
        confidence=0.95, df=len(values) - 1, loc=np.mean(values), scale=st.sem(values)
    )
    expected_dev = (expected_uci - expected_mean) / expected_mean

    # Assertions
    assert np.isclose(stats.mean, expected_mean), (
        f"Expected mean {expected_mean}, got {stats.mean}"
    )
    assert np.isclose(stats.std, expected_std), (
        f"Expected std dev {expected_std}, got {stats.std}"
    )
    assert np.isclose(stats.lci, expected_lci), (
        f"Expected lower confidence interval {expected_lci}, got {stats.lci}"
    )
    assert np.isclose(stats.uci, expected_uci), (
        f"Expected upper confidence interval {expected_uci}, got {stats.uci}"
    )
    assert np.isclose(stats.deviation, expected_dev), (
        f"Expected deviation {expected_dev}, got {stats.deviation}"
    )


# -----------------------------------------------------------------------------
# ReplicationTabulizer
# -----------------------------------------------------------------------------


def test_tabulizer_initial_state():
    """
    Test that ReplicationTabulizer initializes with empty lists and n = 0.
    """
    tab = ReplicationTabulizer()
    assert tab.n == 0
    # Checks for empty lists (equivalent to len(tab.x_i)==0)
    assert not tab.x_i
    assert not tab.cumulative_mean
    assert not tab.stdev
    assert not tab.lower
    assert not tab.upper
    assert not tab.dev


def test_tabulizer_update():
    """
    Test that update correctly appends new statistical data.
    """
    tab = ReplicationTabulizer()
    mock_results = MagicMock()
    mock_results.x_i = 10
    mock_results.mean = 5.5
    mock_results.std = 1.2
    mock_results.lci = 4.8
    mock_results.uci = 6.2
    mock_results.deviation = 0.1

    tab.update(mock_results)

    assert tab.n == 1
    assert tab.x_i == [10]
    assert tab.cumulative_mean == [5.5]
    assert tab.stdev == [1.2]
    assert tab.lower == [4.8]
    assert tab.upper == [6.2]
    assert tab.dev == [0.1]


def test_tabulizer_summary_table():
    """
    Test that summary_table returns a properly formatted DataFrame.
    """
    tab = ReplicationTabulizer()
    tab.n = 3
    tab.x_i = [10, 20, 30]
    tab.cumulative_mean = [5, 10, 15]
    tab.stdev = [1, 2, 3]
    tab.lower = [3, 8, 13]
    tab.upper = [7, 12, 17]
    tab.dev = [0.1, 0.2, 0.3]

    df = tab.summary_table()

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 3
    assert df["Mean"].tolist() == [10, 20, 30]
    assert df["Cumulative Mean"].tolist() == [5, 10, 15]
    assert df["Standard Deviation"].tolist() == [1, 2, 3]
    assert df["Lower Interval"].tolist() == [3, 8, 13]
    assert df["Upper Interval"].tolist() == [7, 12, 17]
    assert df["% deviation"].tolist() == [0.1, 0.2, 0.3]


# -----------------------------------------------------------------------------
# confidence_interval_method
# -----------------------------------------------------------------------------


@pytest.mark.parametrize(
    "input_type",
    [
        pd.Series([1, 2, 3, 4, 5]),
        [1, 2, 3, 4, 5],
        pd.DataFrame({"metric_a": [1, 2, 3, 4, 5], "metric_b": [2, 3, 4, 5, 6]}),
        {"metric_a": [1, 2, 3, 4, 5]},
        [[1, 2, 3, 4, 5], [2, 3, 4, 5, 6]],
    ],
)
def test_confidence_interval_various_inputs(input_type):
    """
    Check the output type from different inputs to confidence_interval_method.
    """
    warnings.filterwarnings(
        "ignore", message="WARNING: the replications do not reach desired precision"
    )
    result = confidence_interval_method(
        replications=input_type, alpha=0.05, desired_precision=0.1, min_rep=3
    )
    if isinstance(input_type, (pd.Series, list)) and not (
        isinstance(input_type[0], (list, np.ndarray, pd.Series))
    ):
        # Single metric returns tuple
        assert isinstance(result, tuple)
        n_reps, results_df = result
        assert isinstance(n_reps, int)
        assert isinstance(results_df, pd.DataFrame)
    else:
        # Multiple metrics returns dict
        assert isinstance(result, dict)
        for val in result.values():
            assert isinstance(val, tuple)
            n_reps, results_df = val
            assert isinstance(n_reps, int)
            assert isinstance(results_df, pd.DataFrame)
