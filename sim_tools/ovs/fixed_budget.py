"""

Fixed budget algorithms

1. OCBA - optimal computing budget allocation
2. OCBA-m - optimal computering budget allocation top m.

"""

import numpy as np


class OCBAM:
    """
    Optimal Computer Budget Allocation Top M (OCBA-m)

    Given a total replication budget T allocate
    replications across designs in order to maximise the
    approximate probability of correctly selecting the best m designs

    Assumes each system design has similar run time.

    Algorithm described in:
    -------------------------


    """

    def __init__(self, model, n_designs, budget, delta, n_0=5, m=2, obj="min"):
        """
        Constructor method for Optimal Budget Computer Allocation Top M

        Parameters:
        ---------

        model - object, simulation model that implements
        interface action(design)

        n_designs - int, (k) the number of competing system designs

        budget - int, (T) the total simulation budget available i.e. the
        total number of replications available to allocate between systems

        delta - int, incremental budget to allocate.  Recommendation is
        > 5 and smaller than 10 percent of budget.  When simulation is
        expensive then this number could be set to 1.

        n_0 - int, the total number of initial replications.  Minimum allowed
        is 5 (default=5)

        m - int, the best m designs

        obj - str, 'min' if minimisation; 'max' if maximisation.
        (default='min')

        """
        model.register_observer(self)
        self._env = model
        self._k = n_designs
        self._T = budget
        self._delta = delta
        self._n_0 = n_0
        self._m = m
        self._obj = obj

        self._allocations = np.zeros(n_designs, np.int32)
        self._means = np.zeros(n_designs, np.float64)
        self._vars = np.zeros(n_designs, np.float64)
        self._ses = np.zeros(n_designs, np.float64)
        self._ratios = np.zeros(n_designs, np.float64)

        # used when calculating running standard deviation across designs
        # sum of squares
        self._sq = np.zeros(n_designs, np.float64)

        if self._obj == "min":
            self._negate = -1.0
            self._min = True
        else:
            self._negate = 1.0
            self._min = False

    def __str__(self):
        return (
            f"OCBA(n_designs={self._k}, m={self._m}, budget={self._T}, "
            + f"delta={self._delta}, n_0={self._n_0}, obj={self._obj})"
        )

    def solve(self):
        """
        This works okay for maximisation, but not if I
        include negatation! What am I doing wrong!
        """

        new_allocations = np.full(shape=self._k, fill_value=self._n_0, dtype=np.int16)

        while self._allocations.sum() < self._T:
            # simulate systems using new allocation of budget
            self._simulate(new_allocations)

            # calculate parameter c and deltas
            c = self._parameter_c(self._k, self._m)

            deltas = self._means - c

            # temp
            # deltas *= self._negate

            # allocate
            new_allocations = np.full(shape=self._k, fill_value=0, dtype=np.int16)

            for _ in range(self._delta):
                values = np.divide(
                    self._allocations + new_allocations,
                    np.square(np.divide(self._ses, deltas)),
                )
                ranks = get_ranks(values)
                new_allocations[ranks.argmin()] += 1

        # return top m
        best = np.argpartition(self._means, -self._m)[-self._m :]

        return best

    def _simulate(self, new_allocations):
        """
        For each design run n_0 initial replications
        """
        for design in range(self._k):
            for _ in range(new_allocations[design]):
                self._env.simulate(design)

    def _parameter_c(self, k, m):
        order = np.argsort(self._means)
        s_ses = self._ses[order]
        s_means = self._means[order]

        return (
            (s_ses[k - m + 1] * s_means[k - m]) + (s_ses[k - m] * s_means[k - m + 1])
        ) / (s_ses[k - m] + s_ses[k - m + 1])

    def feedback(self, *args):
        """
        Feedback from the environment
        Recieves a reward and updates understanding
        of an arm

        Parameters:
        ------
        *args -- list of argument
                 0  sender object
                 1. arm index to update
                 2. observation

        """
        design_index = args[1]
        observation = args[2]
        self._allocations[design_index] += 1

        # update running mean and standard deviation
        self._update_moments(design_index, observation)

    def _update_moments(self, design_index, observation):
        """
        Updates the running average, var of the design

        Parameters:
        ------
        design_index -- int, index of the array to update

        observation -- float, observation recieved from the last replication
        """
        n = self._allocations[design_index]
        current_mean = self._means[design_index]
        new_mean = ((n - 1) / float(n)) * current_mean + (
            1 / float(n)
        ) * observation * self._negate

        if n > 1:
            self._sq[design_index] += (observation - abs(current_mean)) * (
                observation - abs(new_mean)
            )
            self._vars[design_index] = self._sq[design_index] / (n - 1)
            self._ses[design_index] = np.sqrt(self._vars[design_index]) / np.sqrt(n)

        self._means[design_index] = new_mean


def get_ranks(array):
    """
    Returns a numpy array containing ranks of numbers within a input array.
    e.g. [3, 2, 1] returns [2, 1, 0]
    e.g. [3, 2, 1, 4] return [2, 1, 0, 3]

    Parameters:
    --------
    array - np.ndarray, numpy array (only tested with 1d)

    Returns:
    ---------
    np.ndarray, ranks

    """
    temp = array.argsort()
    ranks = np.empty_like(temp)
    ranks[temp] = np.arange(len(array))
    return ranks


class OCBA:
    """
    Optimal Computer Budget Allocation (OCBA)

    Given a total replication budget T allocate
    replications across designs in order to maximise the
    probability of correct selections.

    Assumes each system design has similar run time.

    Algorithm described in:
    -------------------------
    Stochastic Simulation Optimization: An Optimal Computer Budget
    Allocation.  by Chun-Hung Chen and Loo Hay Lee.

    Pages 49-50 for algorithm description
    Page 215 - for example C code.

    """

    def __init__(self, model, n_designs, budget, delta, n_0=5, obj="min"):
        """
        Constructor method for Optimal Budget Computer Allocation

        Parameters:
        ---------

        model - object, simulation model that implements
        interface simulate(design)

        n_designs - int, (k) the number of competing system designs

        budget - int, (T) the total simulation budget available i.e. the
        total number of replications available to allocate between systems

        delta - int, incremental budget to allocate.  Recommendation is
        > 5 and smaller than 10 percent of budget.  When simulation is
        expensive then this number could be set to 1.

        n_0 - int, the total number of initial replications.  Minimum allowed
        is 5 (default=5)

        obj - str, 'min' if minimisation; 'max' if maximisation.
        (default='min')

        """
        if n_0 < 5:
            raise ValueError("n_0 must be >= 5")

        if (budget - (n_designs * n_0)) % delta != 0:
            raise ValueError("(budget - (n_designs * n_0)) must be multiple of delta")

        types = ["min", "max"]
        if obj not in types:
            raise ValueError("obj parameter must be min or max")

        model.register_observer(self)
        self._env = model
        self._k = n_designs
        self._T = budget
        self._delta = delta
        self._n_0 = n_0
        self._obj = obj

        self._allocations = np.zeros(n_designs, np.int32)
        self._means = np.zeros(n_designs, np.float64)
        self._vars = np.zeros(n_designs, np.float64)
        self._ratios = np.zeros(n_designs, np.float64)

        # used when calculating running standard deviation across designs
        # sum of squares
        self._sq = np.zeros(n_designs, np.float64)

        if obj == "min":
            self._negate = 1.0
        else:
            self._negate = -1.0

    def __str__(self):
        return (
            f"OCBA(n_designs={self._k}, budget={self._T}, "
            + f"delta={self._delta}, n_0={self._n_0}, obj={self._obj})"
        )

    def reset(self):
        """
        Reset values
        """
        # self._total_reward = 0
        # self._current_round = 0
        self._allocations = np.zeros(self._k, np.int32)
        self._means = np.zeros(self._k, np.float64)
        self._vars = np.zeros(self._k, np.float64)
        self._ratios = np.zeros(self._k, np.float64)
        self._sq = np.zeros(self._k, np.float64)

    def solve(self):
        """
        run the ranking and selection procedure
        Vanilla OCBA
        """
        self.reset()
        self._initialise()

        while self._allocations.sum() < self._T:
            new_allocations = self._allocate()

            for design in range(self._k):
                for _ in range(new_allocations[design]):
                    self._env.simulate(design)

        best = np.argmin(self._means)
        return best

    def _initialise(self):
        """
        For each design run n_0 initial replications
        """
        for design in range(self._k):
            for _ in range(self._n_0):
                self._env.simulate(design)

    def _allocate(self):
        """
        Allocate the incremental budget across
        designs
        """

        # Notes could do with some cleaning and
        # seperation into functions

        # total allocated + delta
        budget_to_allocate = self._allocations.sum() + self._delta

        # get indicies of best and second best designs so far
        # note treated as minimisation problem.
        # Means are negated if maximisation
        ranks = get_ranks(self._means)
        best_index, s_best_index = np.argpartition(ranks, 2)[:2]

        self._ratios[s_best_index] = 1.0

        # Part 1: Ratio N_i / N_s
        # 'select' just excludes best and second best from arraywise calcs
        select = [i for i in range(self._k) if i not in [best_index, s_best_index]]

        temp = (self._means[best_index] - self._means[s_best_index]) / (
            self._means[best_index] - self._means[select]
        )

        self._ratios[select] = np.square(temp) * (
            self._vars[select] / self._vars[s_best_index]
        )

        # Part 2: N_b
        # exclude best
        select = [i for i in range(self._k) if i not in [best_index]]
        temp = (np.square(self._ratios[select]) / self._vars[select]).sum()
        self._ratios[best_index] = np.sqrt(self._vars[best_index] * temp)

        # got all of the ratios now...
        more_runs = np.full(self._k, True, dtype=bool)
        additional_runs = np.zeros(self._k, dtype="float")
        more_alloc = True

        # do i need to use the more alloc or can i just use mask?
        while more_alloc:
            more_alloc = False

            ratio_s = (more_runs * self._ratios).sum()

            additional_runs[more_runs] = (budget_to_allocate / ratio_s) * self._ratios[
                more_runs
            ]

            # additional_runs = additional_runs.astype(int)
            additional_runs = np.around(additional_runs).astype(int)

            mask = additional_runs < self._allocations
            additional_runs[mask] = self._allocations[mask]

            # disable designs where new allocation is less than has already
            # been run.
            more_runs[mask] = 0

            if mask.sum() > 0:
                more_alloc = True

            if more_alloc:
                budget_to_allocate = self._allocations.sum() + self._delta
                budget_to_allocate -= (additional_runs * ~more_runs).sum()

        total_additional = additional_runs.sum()
        additional_runs[best_index] += (
            self._allocations.sum() + self._delta - total_additional
        )

        return additional_runs - self._allocations

    def feedback(self, *args):
        """
        Feedback from the environment
        Recieves a reward and updates understanding
        of an arm

        Parameters:
        ------
        *args -- list of argument
                 0  sender object
                 1. arm index to update
                 2. observation
        """
        design_index = args[1]
        observation = args[2]
        self._allocations[design_index] += 1

        # update running mean and standard deviation
        self._update_moments(design_index, observation)

    def _update_moments(self, design_index, observation):
        """
        Updates the running average, var of the design

        Parameters:
        ------
        design_index -- int, index of the array to update

        observation -- float, observation recieved from the last replication
        """
        n = self._allocations[design_index]
        current_mean = self._means[design_index]
        new_mean = ((n - 1) / float(n)) * current_mean + (
            1 / float(n)
        ) * observation * self._negate

        if n > 1:
            self._sq[design_index] += (observation - abs(current_mean)) * (
                observation - abs(new_mean)
            )
            self._vars[design_index] = self._sq[design_index] / (n - 1)

        self._means[design_index] = new_mean
