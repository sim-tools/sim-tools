"""Replication back tests

Back testing for code related to selecting the number of replications.

Credit: Some of these tests are adapted from-
    Heather, A. Monks, T. (2025). Python DES RAP Template. GitHub.
    https://github.com/pythonhealthdatascience/rap_template_python_des.
"""

import pytest

from tests.dummy_model import DummySimulationModel
from sim_tools.output_analysis import ReplicationsAlgorithm, confidence_interval_method


@pytest.mark.parametrize(
    "mean, std_dev, expected_n_reps", [(7, 0.2, 5), (10, 2, 58), (50, 1, 4)]
)
def test_ci_method(mean, std_dev, expected_n_reps):
    """
    Use DummySimulationModel to check that results from
    confidence_interval_method are consistent with those previously generated.

    Parameters
    ----------
    mean: float
        Mean for the normal distribution in the dummy model.
    std_dev: float
        Standard deviation for the normal distribution in the dummy model.
    expected_n_reps: int
        Expected result from confidence_interval_method.
    """
    model = DummySimulationModel(mean=mean, std_dev=std_dev)

    replications = [model.single_run(rep)["metric"] for rep in range(1, 101)]

    n_reps, _ = confidence_interval_method(
        replications=replications,
        alpha=0.05,
        desired_precision=0.05,
        min_rep=3,
        decimal_places=2,
    )

    assert n_reps == expected_n_reps, (
        f"Expected {expected_n_reps} replications but got {n_reps}"
    )


@pytest.mark.parametrize(
    "mean, std_dev, expected_n_reps", [(100, 4, 3), (1, 1, 332), (44, 27, 171)]
)
def test_algorithm(mean, std_dev, expected_n_reps):
    """
    Use DummySimulationModel to check that results from ReplicationsAlgorithm
    are consistent with those previously generated.

    Parameters
    ----------
    mean: float
        Mean for the normal distribution in the dummy model.
    std_dev: float
        Standard deviation for the normal distribution in the dummy model.
    expected_n_reps: int
        Expected result from ReplicationsAlgorithm.
    """
    model = DummySimulationModel(mean=mean, std_dev=std_dev)

    analyser = ReplicationsAlgorithm(
        alpha=0.05,
        half_width_precision=0.1,
        initial_replications=5,
        look_ahead=10,
        replication_budget=500,
        verbose=False,
    )

    n_reps, _ = analyser.select(model, metrics=["metric"])

    assert n_reps["metric"] == expected_n_reps, (
        f"Expected {expected_n_reps} replications but got {n_reps['metric']}"
    )
