"""
Toy models for testing optimisation via simulation procedures and
for running models.
"""

import numpy as np


# pylint: disable=too-many-instance-attributes
class ManualOptimiser:
    """
    A class to manually run individual and multiple replications
    from competiting simulated designs of a system.
    """

    def __init__(self, model, n_designs, verbose=False):
        """
        Constructor

        Parameters:
        -------
        model - object, simulation model that implements simulate(design_index)
        and register_observer(observer) methods

        n_designs - int, number of competiting designs

        verbose_replications - bool, True if each individual rep observation
        is displayed.  False if run silently.  (default=False)
        """
        self._model = model
        model.register_observer(self)
        self.verbose = verbose
        self._n_designs = n_designs
        self.means = np.zeros(n_designs, dtype="float")
        self.vars = np.zeros(n_designs, dtype="float")
        self._sq = np.zeros(n_designs, dtype="float")
        self._ses = np.zeros(n_designs, dtype="float")
        self.allocations = np.zeros(n_designs, np.int32)

    def __str__(self):
        return (
            f"ManualOptimiser(model={self._model.__str__()}, "
            + f"n_designs={self._n_designs}, verbose={self.verbose})"
        )

    def simulate_designs(self, design_indexes=None, replications=1):
        """
        Multiple replications of a simulated design

        Parameters:
        --------
        design_indexes - list, zero based design indexes.  If None
        then all system designs are included. (default=None)

        replications - int, number of replications to run (default=1)
        """

        if design_indexes is None:
            design_indexes = list(range(self._n_designs))

        for design_index in design_indexes:
            self.simulate_design(design_index, replications)

    def simulate_design(self, design_index, replications=1):
        """
        Multiple replications of a simulated design

        Parameters:
        --------
        design_index - int, zero based design index
        replications - int, number of replications to run (default=1)
        """
        for _ in range(replications):
            self._model.simulate(design_index)

    def feedback(self, *args):
        """
        Feedback from the simulation model
        Recieves a reward and updates understanding
        of an arm

        Keyword arguments:
        ------
        *args -- list of argument
                 0  sender object
                 1. design index
                 2. observation
        """
        design_index = args[1]
        observation = args[2]

        self.allocations[design_index] += 1
        self._update_moments(design_index, observation)

        if self.verbose:
            print(observation)

    def _update_moments(self, design_index, observation):
        """
        Updates the running average, var of the design

        Parameters:
        ------
        design_index -- int, index of the array to update

        observation -- float, observation recieved from the last replication
        """
        n = self.allocations[design_index]
        current_mean = self.means[design_index]
        new_mean = ((n - 1) / float(n)) * current_mean + (1 / float(n)) * observation

        if n > 1:
            self._sq[design_index] += (observation - abs(current_mean)) * (
                observation - abs(new_mean)
            )
            self.vars[design_index] = self._sq[design_index] / (n - 1)
            self._ses[design_index] = np.sqrt(self.vars[design_index]) / np.sqrt(n)

        self.means[design_index] = new_mean


def gaussian_sequence_model(start, end, step=1):
    """
    Sequence of GaussianBandit Arms.  Assumes unit variance.
    e.g. start = 1, end = 10 return 10 bandits arms with means 1 through 10.

    Parameters:
    -------
    start: int,
        start of sequence of means (inclusive)

    end: int,
        end of sequence of means (inclusive)

    step: int, optional (default = 1)
        step size for sequence

    Returns:
    -------
    list of GaussianBandit objects ordered by means
    """
    bandits = [GaussianBandit(mean) for mean in range(start, end + 1, step)]
    return BanditCasino(bandits)


def gaussian_bandit_sequence(start, end, step=1):
    """
    Sequence of GaussianBandit Arms.  Assumes unit variance.
    e.g. start = 1, end = 10 return 10 bandits arms with means 1 through 10.

    Parameters:
    -------
    start - int, start of sequence of means (inclusive)
    end - int, end of sequence of means (exclusive)
    step - int, step size for sequence (default = 1)

    Returns:
    -------
    list of GaussianBandit objects ordered by means

    """
    return [GaussianBandit(mean) for mean in range(start, end, step)]


def random_gaussian_model(mean_low, mean_high, var_low, var_high, n_designs):
    """
    Create a model with n system designs where the mean and variance of the
    normal distributions are sampled to between the specified tolerances

    Parameters:
    -------
    mean_low - float, a lower bound on the means of the output distributions

    mean_high- float, an upper bound on the means

    var_low - float, a lower bound on the variance of the output distributions

    var_high - float, an upper bound on the variances.

    n_designs - int, the number of designs to create.

    Returns:
    --------
    BanditCasino with n_designs with means and variances between
    specified limits.
    """
    means = np.random.uniform(low=mean_low, high=mean_high, size=n_designs)
    sigmas = np.random.uniform(low=var_low, high=var_high, size=n_designs)
    return custom_gaussian_model(means, sigmas)


def custom_gaussian_model(mus, sigmas):
    """
    Creates a simulation model where each
    output distribution is distributed N ~(mu, sigma)

    Assumes mus and signmas are of equal length

    Keyword arguments:
    ------
    mus - variable size list of means
    sigmas - list, variances

    Returns:
    ------
    object, simulation model
    """
    bandits = [GaussianBandit(mu, sigma) for mu, sigma in zip(mus, sigmas)]
    return BanditCasino(bandits)


class GaussianBandit:
    """
    Classic one armed bandit gambling machine.

    A user plays the bandit by pulling its arm.

    The bandit returns a reward normally distribution
    with mean mu and stdev sigma
    """

    def __init__(self, mu, sigma=1.0):
        """
        Constructor method for Gaussian Bandit

        Keyword arguments:
        -----
        mu -- float, mean of the normal distribution
        sigma -- float, stdev of the normal distribution (default = 1.0)

        """
        self._mu = mu
        self._sigma = sigma
        self._number_of_plays = 0
        self._total_reward = 0
        self._observers = []

    def play(self):
        """
        Pull the arm on the bernoulli bandit


        Returns:
        -----
        reward -- int with value = 0 when no reward
                  and 1 when the pull results in a win
        """
        reward = np.random.normal(self._mu, self._sigma)

        self._total_reward += reward
        self._number_of_plays += 1

        return reward

    def reset(self):
        """
        Reset the number of plays and
        total rewards to zero.
        """
        self._number_of_plays = 0
        self._total_reward = 0


class BanditCasino:
    """
    Bandit Casino
    """

    def __init__(self, bandits):
        """
        Casino constructor method

        Keyword arguments:
        ------
        bandits -- list, of BernoulliBandits objects
        """
        self._bandits = bandits
        self._current_index = 0
        self._observers = []

    def __str__(self):
        return "BanditCasino()"

    def __getitem__(self, index):
        return self._bandits[index]

    def __get_number_of_arms(self):
        return len(self._bandits)

    def _get_best_arm(self):
        means = []
        for bandit in self._bandits:
            means.append(bandit._mu)

        return np.argmax(np.array(means))

    def simulate(self, bandit_index):
        """
        Play a specific bandit machine.

        Notifies all observers of the outcome

        Keyword arguments:
        -----
        bandit_index -- int, index of bandit to play
        """
        reward = self._bandits[bandit_index].play()
        self.notify_observers(bandit_index, reward)

    def random_action(self):
        """
        Selects a bandit index at random and plays it.
        """
        bandit_index = np.random.choice(len(self._bandits))
        self.simulate(bandit_index)

    def __iter__(self):
        return self

    def __next__(self):
        self._current_index += 1
        if self._current_index > len(self._bandits):
            self._current_index = 0
            raise StopIteration
        return self._bandits[self._current_index - 1]

    def register_observer(self, observer):
        """
        Registers an observer to receive notifications.
        """
        self._observers.append(observer)

    def notify_observers(self, *args, **kwargs):
        """
        Notifies all registered observers with the provided arguments.
        """
        for observer in self._observers:
            observer.feedback(self, *args, **kwargs)

    number_of_arms = property(__get_number_of_arms)
    best_design = property(_get_best_arm)
