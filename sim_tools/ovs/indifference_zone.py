"""
Contains algorithms for Optimisation via Simulation
with an indifference zone framework (fixed precision optimisation)

1. KN - (Kim-Nelson) sequential Ranking and Selection Algorithm

2. KNPlusPlus - Kim and Nelson updated their original KN procedure to
update the variance estimate of designs at each stage.

"""

import numpy as np


class KNPlusPlus:
    """
    KN++ algorithm for Ranking and Selection

    More coordiation than KN. V
    Variances are updated at each stage.

    References.
    http://users.iems.northwestern.edu/~nelsonb/Publications/17KimNelson.pdf
    https://www2.isye.gatech.edu/~skim/KimNelson.pdf

    """

    def __init__(self, model, n_designs, delta, alpha=0.05, n_0=2, obj="max"):
        """
        Constructor method for KN++  Ranking and Selection Procedure.

        Reference = 'TO ADD'

        Parameters:
        ----------

        model - object, simulation model that implements
        method simulate(design, replications)

        n_designs - int, the number of designs (k)

        delta - float, the indifference zone

        alpha - float, between 0 and 1.  Used to calculate the 1-alpha
        probability of correct selection

        n_0 - int, the number of initial observations (default = 2)

        obj - str, objective 'max' or 'min' (default='max)

        """
        model.register_observer(self)
        self._env = model

        self._allocations = np.zeros(n_designs, np.int32)
        self._means = np.zeros(n_designs, np.float64)

        # sample variances for each design
        self._vars = np.zeros(n_designs, np.float64)
        # used for calculating running sample variance
        self._sq = np.zeros(n_designs, np.float64)

        # should I also be storing the variances of the differences?

        self._k = n_designs

        # set of non-eliminated systems
        # designs are screened at each stage and removed.
        self._contenders = np.arange(self._k)

        self._delta = delta
        self._alpha = alpha
        self._n_0 = n_0
        self._r = 0

        types = ["min", "max"]
        if obj not in types:
            raise ValueError("obj parameter must be min or max")

        # code set up for max negate for min.
        if obj == "max":
            self._negate = 1.0
        else:
            self._negate = -1.0

        self._obj = obj
        self._n = None
        self._eta = None
        self._h_squared = None
        self._contenders_old = None

    def _calculate_eta(self):
        return 0.5 * (
            np.power(
                2 * (1 - np.power((1 - self._alpha), 1 / (self._k - 1))),
                -2 / (self._n - 1),
            )
            - 1
        )

    def __str__(self):
        return (
            f"KN(n_designs={self._k}, delta={self._delta}, "
            + f"alpha={self._alpha}, n_0={self._n_0}, obj={self._obj})"
        )

    def reset(self):
        """Resets all attributes"""
        # self._total_reward = 0
        self._allocations = np.zeros(self._k, np.int32)
        self._means = np.zeros(self._k, np.float64)
        self._allocations = np.zeros(self._k, np.int32)
        # self._init_obs = np.zeros((self._k, self._n_0), np.float64)
        self._contenders = np.arange(self._k)
        self._vars = np.zeros(self._k, np.float64)
        self._sq = np.zeros(self._k, np.float64)
        self._n = 0

    def solve(self):
        """Run KN++"""
        self.reset()
        self._initialisation()

        while not self._stopping():
            self._screening()
            self._sequential_replication()
            self._update()

        return self._contenders

    def _initialisation(self):
        """
        Initialise KN++

        Estimate initial sample means and variances of the differences

        """
        for _ in range(self._n_0):
            self._sequential_replication()

        self._eta = self._calculate_eta()
        self._h_squared = 2 * self._eta * (self._n - 1)

    def _stopping(self):
        """
        If |I| = 1 then stop (and select the system whose index is in I)
        else take one further sample from each system

        """
        # Returns True if this is the case and False otherwise
        return len(self._contenders) == 1

    def _sequential_replication(self):
        """
        Run a single replication of each
        design
        """
        for design in self._contenders:
            self._env.simulate(design)

        self._n += 1

    def _screening(self):
        """
        Loop through remaining contenders and screen if
        mean(design_i) < mean(design_j) - epsilon_ij

        where epsilon_ij (w_ij in some papers) determines how far the sample
        mean from system i can drop below the sample mean of system j without
        being eliminated.
        """
        self._contenders_old = np.asarray(self._contenders).copy()
        contenders_mask = np.full(self._contenders.shape, True, dtype=bool)

        # inefficient way to code this...
        # will update to pure numpy at some stage

        for i, design_i in enumerate(self._contenders_old):
            for j, design_j in enumerate(self._contenders_old):
                if i != j:
                    w_ij = self._elimination_distance(design_i, design_j)

                    if self._means[design_i] < self._means[design_j] - w_ij:
                        contenders_mask[i] = False
                        break

        self._contenders = self._contenders[contenders_mask]

    def _mean_difference(self, design_i, design_j):
        """diff between two sample means"""
        mean_i = self._means[design_i]
        mean_j = self._means[design_j]
        return mean_i - mean_j

    def _elimination_distance(self, design_i, design_j):
        """
        Returns the distance mean(design_i) can fall below mean(design_j)
        before it is ELIMINATED.

        Parameters:
        -----
        design_i - int, the index of the first design

        design_j - int, the index of the second design

        Returns:
        --------
        float.
        """
        # sample variances
        var_i = self._vars[design_i]
        var_j = self._vars[design_j]
        sum_of_vars = var_i + var_j

        # I think the problem is that I am not using the variance of the
        # differences.
        return (self._delta / (2 * self._n)) * (
            ((self._h_squared * (sum_of_vars)) / self._delta**2) - self._n
        )
        # return max(0, w_ij)

    def _update(self):
        """update step.
        Recaulcate eta and h_squared
        """
        self._eta = self._calculate_eta()
        self._h_squared = 2 * self._eta * (self._n - 1)

    def feedback(self, *args):
        """
        Feedback from the simulated environment
        Recieves an observation from the system

        Parameters:
        ------
        *args -- list of argument
                 0  sender object
                 1. design index to update
                 2. observation
        """
        design_index = args[1]
        observation = args[2]
        self._allocations[design_index] += 1

        # update running mean and standard deviation of the design
        self._update_sample_estimates(design_index, observation)

    def _update_sample_estimates(self, design_index, observation):
        """
        Updates the running sample mean and var of the design

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


class KN:
    """
    KN - (Kim-Nelson) sequential Ranking and Selection Algorithm
    """

    def __init__(self, model, n_designs, delta, alpha=0.05, n_0=2, obj="max"):
        """
        Constructor method for KN  Ranking and Selection Procedure.

        This works well for up to 20 competing designs

        References.
        http://users.iems.northwestern.edu/~nelsonb/Publications/KimNelsonKN.pdf

        Parameters:
        ----------

        model - object, simulation model that implements
        method simulate(design, replications)

        n_designs - int, the number of designs (k)

        delta - float, the indifference zone

        alpha - float, between 0 and 1.  Used to calculate the 1-alpha
        probability of correct selection

        n_0 - int, the number of initial observations (default = 2)

        """
        model.register_observer(self)
        self._env = model

        self._current_round = 0
        self._allocations = np.zeros(n_designs, np.int32)
        self._means = np.zeros(n_designs, np.float64)
        self._init_obs = np.zeros((n_designs, n_0), np.float64)

        self._k = n_designs

        # the set of non-eliminated systems I
        self._contenders = np.arange(self._k)

        self._delta = delta
        self._alpha = alpha
        self._n_0 = n_0

        # number of replications that has been run.
        self._r = 0

        self._eta = 0.5 * (
            np.power((2 * alpha) / (self._k - 1), -2 / (self._n_0 - 1)) - 1
        )

        types = ["min", "max"]
        if obj not in types:
            raise ValueError("obj parameter must be min or max")

        # code set up for max negate for min.
        if obj == "max":
            self._negate = 1.0
        else:
            self._negate = -1.0

        self._obj = obj
        self._h_squared = None
        self._contenders_old = None
        self._variance_of_diffs = None

    def __str__(self):
        return (
            f"KN(n_designs={self._k}, delta={self._delta}, "
            + f"alpha={self._alpha}, n_0={self._n_0}, obj={self._obj})"
        )

    def reset(self):
        """
        Reset values.
        """
        # self._total_reward = 0
        self._allocations = np.zeros(self._k, np.int32)
        self._means = np.zeros(self._k, np.float64)
        self._allocations = np.zeros(self._k, np.int32)
        self._init_obs = np.zeros((self._k, self._n_0), np.float64)
        self._contenders = np.arange(self._k)
        self._r = 0

    def solve(self):
        """Run procedure KN"""
        self.reset()
        self._initialisation()

        while not self._stopping():
            self._screening()
            self._sequential_replication()

        return self._contenders

    def _initialisation(self):
        """
        Initialise KN

        Estimate initial sample means and variances of the differences

        """
        self._h_squared = 2 * self._eta * (self._n_0 - 1)

        for _ in range(self._n_0):
            self._sequential_replication()

        self._variance_of_diffs = self._calc_variance_of_differences()

    def _sequential_replication(self):
        for design in self._contenders:
            self._env.simulate(design)

        self._r += 1

    def _calc_variance_of_differences(self):
        pairwise_diffs = self._init_obs[:, None] - self._init_obs
        variance_of_diffs = pairwise_diffs.var(axis=-1, ddof=1)
        # flattens array and drops differences with same design
        return variance_of_diffs[~np.eye(variance_of_diffs.shape[0], dtype=bool)]

    # need to check if this is correct...
    # possibly a bug.  Works when in designs are ordered, but not otherwise...
    # think this is fixed 28/02/2020 TM - to double check.
    def _screening(self):
        self._contenders_old = np.asarray(self._contenders).copy()
        contenders_mask = np.full(self._contenders.shape, True, dtype=bool)
        # terrible way to code this...!

        # designs in contention for this round
        for i, design_i in enumerate(self._contenders_old):
            for j, design_j in enumerate(self._contenders_old):
                if i != j and not self._design_still_in_contention(design_i, design_j):
                    contenders_mask[i] = False
                    break

        self._contenders = self._contenders[contenders_mask]

    def _design_still_in_contention(self, design_i, design_j):
        w_il = self._limit_to_distance_from_sample_means(design_i, design_j)

        mean_i = self._means[design_i]
        mean_l = self._means[design_j]

        return mean_i >= mean_l - w_il

    def _limit_to_distance_from_sample_means(self, design_i, design_l):
        """
        calculates W_li(r),
        which determines how far the sample mean from system i can
        drop below the sample means of system l without being eliminated
        """
        # note - variance_of_diffs is a flat array for all comparisons and
        # requires this formaula to look up the value.
        index = design_i * (self._k - 1) + (design_l - 1)

        w_il = (self._delta / (2 * self._r)) * (
            ((self._h_squared * self._variance_of_diffs[index]) / self._delta**2)
            - self._r
        )

        return max(0, w_il)

    def _stopping(self):
        """
        If |I| = 1 then stop (and select the system whose index is in I)
        else take one further sample from each system

        """
        # Returns True if this is the case and False otherwise
        return len(self._contenders) == 1

    def feedback(self, *args):
        """
        Feedback from the environment
        Recieves a reward and updates understanding
        of an arm

        Keyword arguments:
        ------
        *args -- list of argument
                 0  sender object
                 1. design index to update
                 2. observation
        """
        design_index = args[1]
        observation = args[2]
        self._allocations[design_index] += 1
        self._means[design_index] = self.updated_mean_estimate(
            design_index, observation
        )
        if self._r < self._n_0:
            self._init_obs[design_index][self._r] = observation

    def updated_mean_estimate(self, design_index, observation):
        """
        Calculate the new running average of the design

        Keyword arguments:
        ------
        design_index -- int, index of the array to update
        observation -- float, observation recieved from the last action

        Returns:
        ------
        float, the new mean estimate for the selected arm
        """
        n = self._allocations[design_index]
        current_value = self._means[design_index]

        return ((n - 1) / float(n)) * current_value + (
            1 / float(n)
        ) * observation * self._negate
