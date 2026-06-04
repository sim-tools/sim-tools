"""Dummy simulation model used for testing"""

import numpy as np


# pylint: disable=too-few-public-methods
class DummySimulationModel:
    """
    Dummy simulation model used for testing.
    """

    def __init__(self, mean=100, std_dev=5):
        """
        Initialises the dummy model.

        Parameters
        ----------
        mean: float, optional (default=100)
            Mean for the normal distribution.
        std_dev: float, optional (default=5)
            Standard deviation for the normal distribution.
        """
        self.mean = mean
        self.std_dev = std_dev

    def single_run(self, replication_number: int) -> dict[str, float]:
        """
        Simulate a single replication with controlled randomness.

        Parameters
        ----------
        replication_number: int
            Dummy replication number, used as random seed when sampling from
            the normal distribution.

        Returns
        -------
        dict[str, float]
            {"metric": simulated_value}
        """
        value = np.random.default_rng(replication_number).normal(
            loc=self.mean, scale=self.std_dev
        )
        return {"metric": value}
