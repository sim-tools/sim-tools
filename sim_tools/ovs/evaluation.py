"""
module evaluation

Provides:

* Monte carlo classes and function to test of R&S procedures in
simulated environments environments

"""

import numpy as np
import pandas as pd
from sklearn.model_selection import ParameterGrid


# pylint: disable=too-few-public-methods
class AgentSimulation:
    """
    Encapsulates a Monte-Carlo simulation framework for agents in
    multi-arm bandit environment.

    Agents must implement the interface

    solve()
    reset()
    """

    def __init__(self, environment, agent, replications=1000):
        """
        Initialises the AgentSimulation object with the given environment,
        agent, and parameters.
        """
        self._env = environment
        self._agent = agent
        self._reps = replications

    def simulate(self):
        """
        Runs the Monte-Carlo simulation for the specified number of
        replications.
        """
        best_indexes = np.zeros(self._reps, np.int32)

        for rep in range(self._reps):
            self._agent.reset()
            best_indexes[rep] = self._agent.solve()
            # best_indexes[rep] = self._agent.best_arm

        return best_indexes


# pylint: disable=too-few-public-methods
class ExperimentResults:
    """
    Results Container for an Agent Experiment
    """

    def __init__(
        self, selections, correct_selections, p_correct_selections, opportunity_cost
    ):
        """
        Initialises the ExperimentResults object with the given results.
        """
        self.selections = selections
        self.correct_selections = correct_selections
        self.p_correct_selections = p_correct_selections
        self.expected_opportunity_cost = opportunity_cost


class Experiment:
    """
    Tests the power of a given configuration of an agent to select the best
    design in an environment based on a specific objective ("max" or "min").
    """

    # pylint: disable=too-many-arguments,too-many-positional-arguments
    def __init__(
        self, env, procedure, best_index=0, objective="max", replications=1000
    ):
        """
        Initialises the Experiment object with the given environment, agent,
        and parameters.
        """
        self._env = env
        self._agent = procedure
        self._sim = AgentSimulation(env, agent=procedure, replications=replications)
        self._best_index = best_index
        self._objective = objective
        self._reps = replications

    def execute(self):
        """
        Execute the experiment
        """
        selections = self._sim.simulate()

        correct_selections = (selections == self._best_index).sum()

        p_correct_selections = correct_selections / self._reps

        opportunity_cost = self._calculate_exp_opportunity_cost(selections)

        return ExperimentResults(
            selections, correct_selections, p_correct_selections, opportunity_cost
        )

    def _calculate_exp_opportunity_cost(self, selections):
        """
        Calculates the opportunity cost based on the agent's selections
        compared to the best design.
        """
        best_mean = self._env[self._env.best_design]._mu

        oc = 0.0
        for index in selections:
            oc += best_mean - self._env[index]._mu
        return oc / len(selections)


class GridExperiment:
    """
    Performs grid search search-based experiments by varying the parameters of
    an agent and executing experiments for each configuration.
    """

    # pylint: disable=too-many-arguments,too-many-positional-arguments
    def __init__(self, agent, environment, param_grid, best_index=0, replications=1000):
        """
        Initialises the GridExperiment with the given agent, environment, and
        parameter grid.
        """
        self._agent = agent
        self._env = environment
        self._param_grid = ParameterGrid(param_grid)
        self._replications = replications
        self._best_index = best_index

    def fit(self):
        """
        Fits the agent to the environment by running experiments for all
        parameter configurations in the parameter grid. The results are stored
        in a DataFrame.
        """
        # create data frame to store results
        num_rows = len(self._param_grid)
        columns = list(self._param_grid[0].keys())
        columns.append("correct_selections")
        columns.append("p_correct_selections")

        df_results = pd.DataFrame(index=np.arange(0, num_rows), columns=columns)

        # simulate each of the agent configurations
        for index, param_dict in enumerate(self._param_grid):
            for key, value in param_dict.items():
                # set the agents attribute
                setattr(self._agent, key, value)

                df_results.loc[index][key] = value

            experiment = Experiment(
                self._env,
                self._agent,
                best_index=self._best_index,
                replications=self._replications,
            )

            results = experiment.execute()

            df_results.loc[index]["correct_selections"] = results.correct_selections
            df_results.loc[index]["p_correct_selections"] = results.p_correct_selections

        return df_results
