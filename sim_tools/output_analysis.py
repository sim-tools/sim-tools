"""
module: output_analysis

Provides tools for selecting the number selecting the number of
replications to run with a Discrete-Event Simulation.

The Confidence Interval Method (tables and visualisation)

The Replications Algorithm (Hoad et al. 2010).
"""

import warnings
from collections.abc import Callable, Sequence
from typing import (
    Any,
    Protocol,
    runtime_checkable,
)

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.stats import t

OBSERVER_INTERFACE_ERROR = (
    "Observers of OnlineStatistics must implement "
    + "ReplicationObserver interface. i.e. "
    + "update(results: OnlineStatistics) -> None"
)

ALG_INTERFACE_ERROR = (
    "Parameter 'model' must implement "
    + "ReplicationsAlgorithmModelAdapter interface. i.e. "
    + "single_run(replication_no: int) -> float"
)


@runtime_checkable
class ReplicationObserver(Protocol):
    """
    Interface (protocol) for observers that track simulation replication
    results.
    """

    def update(self, results) -> None:
        """
        Add an observation of a replication

        Parameters
        -----------
        results: OnlineStatistic
            The current replication to observe.
        """


@runtime_checkable
class AlgorithmObserver(Protocol):
    """
    Protocol for observer classes used in ReplicationsAlgorithm.

    Classes implementing this protocol should provide a `dev` attribute
    to store observations, a method `update` to add new results, and a
    `summary_table` method to summarize the stored replication statistics.

    Attributes
    ----------
    dev : List[Any]
        Collection of observed replication results.

    Methods
    -------
    update(results) -> None
        Add an observation of a replication.

    summary_table() -> pd.DataFrame
        Create a DataFrame summarising all recorded replication statistics.
    """

    dev: list[Any]

    def update(self, results) -> None: ...

    def summary_table(self) -> pd.DataFrame: ...


class OnlineStatistics:
    """
    Computes running sample mean and variance using Welford's algorithm.

    This is a robust and numerically stable approach first described in the
    1960s and popularised in Donald Knuth's *The Art of Computer Programming*
    (Vol. 2).

    The term *"online"* means each new data point is processed immediately
    to update statistics, without storing or reprocessing the entire dataset.

    This implementation additionally supports computation of:
      - Confidence intervals (CIs).
      - Percentage deviation of CI half-widths from the mean.

    Attributes
    ----------
    n : int
        Number of data points processed so far.
    x_i : float
        Most recent data point.
    mean : float
        Current running mean.
    _sq : float
        Sum of squared differences from the current mean (used for variance).
    alpha : float
        Significance level for confidence interval calculations
    observer : list
        Registered observers notified upon updates.
    """

    def __init__(
        self,
        data: np.ndarray | None = None,
        alpha: float | None = 0.1,
        observer: ReplicationObserver | None = None,
    ) -> None:
        """
        Initialise a new OnlineStatistics object.

        Parameters
        ----------
        data: np.ndarray, optional (default = None)
            Initial dataset to process.

        alpha: float, optional (default = 0.1)
            Significance level for confidence interval calculations
            (CI level = 100 * (1 - alpha) %).

        observer: ReplicationObserver, optional (default=None)
            A user may optionally track the updates to the statistics using a
            `ReplicationObserver` (e.g. `ReplicationTabuliser`). This allows
            further tabular or visual analysis or saving results to file if
            required.

        Raises
        ------
        ValueError
            If `data` is provided but is not a NumPy array.
        """

        self.n = 0
        self.x_i = None
        self.mean = None
        self._sq = None
        self.alpha = alpha
        self._observers = []
        if observer is not None:
            self.register_observer(observer)

        if data is not None:
            if isinstance(data, np.ndarray):
                for x in data:
                    self.update(x)
            # Raise an error if in different format - else will invisibly
            # proceed and won't notice it hasn't done this
            else:
                raise ValueError(f"data must be np.ndarray but is type {type(data)}")

    def register_observer(self, observer: ReplicationObserver) -> None:
        """
        Register an observer to be notified on each statistics update.

        Parameters
        ----------
        observer : ReplicationObserver
            Object implementing the observer interface.

        Raises
        ------
        ValueError
            If `observer` is not an instance of ReplicationObserver.
        """
        if not isinstance(observer, ReplicationObserver):
            raise ValueError(OBSERVER_INTERFACE_ERROR)

        self._observers.append(observer)

    @property
    def variance(self) -> float:
        """
        Sample variance of the data.

        Returns
        -------
        float
            Sample variance, calculated as the sum of squared differences
            from the mean divided by (n - 1).
        """
        return self._sq / (self.n - 1)

    @property
    def std(self) -> float:
        """
        Standard deviation of data.

        Returns
        -------
        float
            Standard deviation, or NaN if fewer than 3 points are available.
        """
        if self.n > 2:
            return np.sqrt(self.variance)
        return np.nan

    @property
    def std_error(self) -> float:
        """
        Standard error of the mean.

        Returns
        -------
        float
            Standard error, equal to `std / sqrt(n)`.
        """
        return self.std / np.sqrt(self.n)

    @property
    def half_width(self) -> float:
        """
        Half-width of the confidence interval.

        Returns
        -------
        float
            The margin of error for the confidence interval.
        """
        dof = self.n - 1
        t_value = t.ppf(1 - (self.alpha / 2), dof)
        return t_value * self.std_error

    @property
    def lci(self) -> float:
        """
        Lower bound of the confidence interval.

        Returns
        -------
        float
            Lower confidence limit, or NaN if fewer than 3 values are
            available.
        """
        if self.n > 2:
            return self.mean - self.half_width
        return np.nan

    @property
    def uci(self) -> float:
        """
        Upper bound of the confidence interval.

        Returns
        -------
        float
            Upper confidence limit, or NaN if fewer than 3 values are
            available.
        """
        if self.n > 2:
            return self.mean + self.half_width
        return np.nan

    @property
    def deviation(self) -> float:
        """
        Precision of the confidence interval expressed as the percentage
        deviation of the half width from the mean.

        Returns
        -------
        float
            CI half-width divided by the mean, or NaN if fewer than 3 values.
        """
        if self.n > 2:
            return self.half_width / self.mean
        return np.nan

    def update(self, x: float) -> None:
        """
        Update statistics with a new observation using Welford's algorithm.

        See Knuth. D `The Art of Computer Programming` Vol 2. 2nd ed. Page 216.

        Parameters
        ----------
        x : float
            New observation.
        """
        self.n += 1
        self.x_i = x
        # Initial statistics
        if self.n == 1:
            self.mean = x
            self._sq = 0
        else:
            # Updated statistics
            updated_mean = self.mean + ((x - self.mean) / self.n)
            self._sq += (x - self.mean) * (x - updated_mean)
            self.mean = updated_mean
        self.notify()

    def notify(self) -> None:
        """
        Notify all registered observers that an update has occurred.
        """
        for observer in self._observers:
            observer.update(self)


class ReplicationTabulizer:
    """
    Observer class for recording replication results from an
    `OnlineStatistics` instance during simulation runs or repeated experiments.

    Implements the observer pattern to collect statistics after each update
    from the observed object, enabling later tabulation and analysis. After
    data collection, results can be exported as a summary dataframe (equivalent
    Implement as the part of observer pattern. Provides a summary frame
    to the output of `confidence_interval_method`).

    Attributes
    ----------
    stdev : list[float]
        Sequence of recorded standard deviations.
    lower : list[float]
        Sequence of recorded lower confidence interval bounds.
    upper : list[float]
        Sequence of recorded upper confidence interval bounds.
    dev : list[float]
        Sequence of recorded percentage deviations of CI half-width from the
        mean.
    cumulative_mean : list[float]
        Sequence of running mean values.
    x_i : list[float]
        Sequence of last observed raw data points.
    n : int
        Total number of updates recorded.
    """

    def __init__(self):
        """
        Initialise an empty `ReplicationTabulizer`.

        All recorded metrics are stored in parallel lists, which grow as
        `update()` is called.
        """
        self.stdev = []
        self.lower = []
        self.upper = []
        self.dev = []
        self.cumulative_mean = []
        self.x_i = []
        self.n = 0

    def update(self, results: OnlineStatistics) -> None:
        """
        Record the latest statistics from an observed `OnlineStatistics`
        instance.

        This method should be called by the observed object when its state
        changes (i.e., when a new data point has been processed).

        Parameters
        ----------
        results : OnlineStatistics
            The current statistics object containing the latest values.
        """
        self.x_i.append(results.x_i)
        self.cumulative_mean.append(results.mean)
        self.stdev.append(results.std)
        self.lower.append(results.lci)
        self.upper.append(results.uci)
        self.dev.append(results.deviation)
        self.n += 1

    def summary_table(self) -> pd.DataFrame:
        """
        Compile all recorded replications into a pandas DataFrame.

        Returns
        -------
        pandas.DataFrame
            A table with one row per replication (update), containing:
            - `Mean` (latest observed value)
            - `Cumulative Mean`
            - `Standard Deviation`
            - `Lower Interval`
            - `Upper Interval`
            - `% deviation` (CI half-width as a fraction of cumulative mean)
        """
        # combine results into a single dataframe
        results = pd.DataFrame(
            [
                self.x_i,
                self.cumulative_mean,
                self.stdev,
                self.lower,
                self.upper,
                self.dev,
            ]
        ).T
        results.columns = [
            "Mean",
            "Cumulative Mean",
            "Standard Deviation",
            "Lower Interval",
            "Upper Interval",
            "% deviation",
        ]
        results.index = np.arange(1, self.n + 1)
        results.index.name = "replications"

        return results


def confidence_interval_method(
    replications: pd.Series
    | pd.DataFrame
    | Sequence[float]
    | Sequence[Sequence[float]]
    | dict[str, Sequence[float]],
    alpha: float | None = 0.05,
    desired_precision: float | None = 0.1,
    min_rep: int | None = 5,
    decimal_places: int | None = 2,
):
    """
    Determine the minimum number of simulation replications required to achieve
    a target precision in the confidence interval of one or several performance
    metrics.

    This function applies the **confidence interval method**: it identifies the
    smallest replication count where the relative half-width of the confidence
    interval is less than the specified `desired_precision` for each metric.

    Parameters
    ----------
    replications: array-like, pd.Series, pd.DataFrame, list, or dict
        Replication results for one or more performance metrics. Accepted
        formats:
        - `pd.Series` or 1D list/numpy array → single metric
        - `pd.DataFrame` → multiple metrics in columns
        - `dict[str, list/array/Series]` → {metric_name: replications}
        - list of lists / numpy arrays / Series → multiple metrics unnamed
        Each inner sequence/Series/numpy array must contain numeric replication
        results in the order they were generated.
    alpha: float, optional (default=0.05)
        Significance level for confidence interval calculations
        (CI level = 100 * (1 - alpha) %).
    desired_precision: float, optional (default=0.1)
        Target CI half-width precision (i.e. percentage deviation of the
        confidence interval from the mean).
    min_rep: int, optional (default=5)
        Minimum number of replications to consider before evaluating precision.
        Helps avoid unstable early results.
    decimal_places: int, optional (default=2)
        Number of decimal places to round values in the returned results table.

    Returns
    -------
    - Single-metric input → tuple `(n_reps, results_df)`
    - Multi-metric input → dict:
        `{metric_name: (n_reps, results_df)}`
    Where:
        n_reps : int
            The smallest number of replications achieving the desired
            precision. Returns -1 if precision is never reached.
        results_df : pandas.DataFrame
            Summary statistics at each replication:
            "Mean", "Cumulative Mean", "Standard Deviation",
            "Lower Interval", "Upper Interval", "% deviation"

    Warns
    -----
    UserWarning
        Issued per metric if the desired precision is never reached.
    """

    def process_single_metric(metric_values):
        """Get result for one metric."""
        # Set up method for calculating statistics
        observer = ReplicationTabulizer()
        stats = OnlineStatistics(
            alpha=alpha, data=np.array(metric_values[:2]), observer=observer
        )

        # Calculate statistics with each replication
        for i in range(2, len(metric_values)):
            stats.update(metric_values[i])

        results_df = observer.summary_table()

        # Find minimum number of replications where deviation is below target
        try:
            n_reps = (
                results_df.iloc[min_rep:]
                .loc[results_df["% deviation"] <= desired_precision]
                .iloc[0]
                .name
            )
        except IndexError:
            msg = "WARNING: the replications do not reach desired precision"
            warnings.warn(msg)
            n_reps = -1

        return n_reps, results_df.round(decimal_places)

    # Single metric
    if isinstance(replications, pd.Series) or np.ndim(replications) == 1:
        return process_single_metric(list(replications))

    # Dataframe with multiple metric columns
    if isinstance(replications, pd.DataFrame):
        return {
            col: process_single_metric(replications[col].tolist())
            for col in replications.columns
        }

    # Dictionary of metrics
    if isinstance(replications, dict):
        return {
            name: process_single_metric(vals) for name, vals in replications.items()
        }

    # List of lists, arrays or series
    if isinstance(replications, list) and all(
        isinstance(x, (list, np.ndarray, pd.Series)) for x in replications
    ):
        return {
            f"metric_{i}": process_single_metric(vals)
            for i, vals in enumerate(replications)
        }

    raise TypeError(f"Unsupported replications type: {type(replications)}")


def plotly_confidence_interval_method(
    n_reps, conf_ints, metric_name, figsize=(1200, 400), shaded=True
):
    """
    Create an interactive Plotly visualisation of the cumulative mean and
    confidence intervals for each replication.

    This plot displays:
      - The running (cumulative) mean of a performance metric.
      - Lower and upper bounds of the confidence interval at each replication.
      - Annotated deviation (as % of mean) on hover.
      - A vertical dashed line at the minimum number of replications (`n_reps`)
        required to achieve the target precision.

    Parameters
    ----------
    n_reps: int
        Minimum number of replications needed to achieve desired precision
        (typically the output of `confidence_interval_method`).
    conf_ints: pandas.DataFrame
        Results DataFrame from `confidence_interval_method`, containing
        columns: `"Cumulative Mean"`, `"Lower Interval"`, `"Upper Interval"`,
        etc.
    metric_name: str
        Name of the performance metric displayed in the y-axis label.
    figsize: tuple, optional (default=(1200,400))
        Figure size in pixels: (width, height).
    shaded: bool, optional
        If True, use shaded CI region. If False, use dashed lines (legacy).

    Returns
    -------
    plotly.graph_objects.Figure
    """
    fig = go.Figure()

    # Calculate relative deviations
    deviation_pct = (
        (conf_ints["Upper Interval"] - conf_ints["Cumulative Mean"])
        / conf_ints["Cumulative Mean"]
        * 100
    ).round(2)

    # Confidence interval
    if shaded:
        # Shaded style
        fig.add_trace(
            go.Scatter(
                x=conf_ints.index,
                y=conf_ints["Upper Interval"],
                mode="lines",
                line={"width": 0},
                name="Upper Interval",
                text=[f"Deviation: {d}%" for d in deviation_pct],
                hoverinfo="x+y+name+text",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=conf_ints.index,
                y=conf_ints["Lower Interval"],
                mode="lines",
                line={"width": 0},
                fill="tonexty",
                fillcolor="rgba(0,176,185,0.2)",
                name="Lower Interval",
                text=[f"Deviation: {d}%" for d in deviation_pct],
                hoverinfo="x+y+name+text",
            )
        )
    else:
        # Dashed lines style
        for col, color, dash in zip(
            ["Lower Interval", "Upper Interval"],
            ["lightblue", "lightblue"],
            ["dot", "dot"],
        ):
            fig.add_trace(
                go.Scatter(
                    x=conf_ints.index,
                    y=conf_ints[col],
                    line={"color": color, "dash": dash},
                    name=col,
                    text=[f"Deviation: {d}%" for d in deviation_pct],
                    hoverinfo="x+y+name+text",
                )
            )

    # Cumulative mean line
    fig.add_trace(
        go.Scatter(
            x=conf_ints.index,
            y=conf_ints["Cumulative Mean"],
            line={"color": "blue", "width": 2},
            name="Cumulative Mean",
            hoverinfo="x+y+name",
        )
    )

    # Vertical threshold line
    fig.add_shape(
        type="line",
        x0=n_reps,
        x1=n_reps,
        y0=0,
        y1=1,
        yref="paper",
        line={"color": "red", "dash": "dash"},
    )

    # Configure layout
    fig.update_layout(
        width=figsize[0],
        height=figsize[1],
        xaxis_title="Replications",
        yaxis_title=f"Cumulative Mean: {metric_name}",
        hovermode="x unified",
        showlegend=True,
    )

    return fig


@runtime_checkable
class ReplicationsAlgorithmModelAdapter(Protocol):
    """
    Adapter pattern for the "Replications Algorithm".

    All models that use ReplicationsAlgorithm must provide a
    single_run(replication_number) interface.
    """

    def single_run(self, replication_number: int) -> dict[str, float]:
        """
        Run a single unique replication of the model and return results.

        Parameters
        ----------
        replication_number : int
            The replication sequence number.

        Returns
        -------
        dict[str, float]
            {'metric_name': value, ... } for all metrics being tracked.
        """


class ReplicationsAlgorithm:
    """
    Automatically determine the number of simulation replications needed
    to achieve and maintain a target confidence interval precision.

    Implements the *Replications Algorithm* from Hoad, Robinson & Davies
    (2010), which combines:
      - The **confidence interval method** to assess whether the
        target precision has been met.
      - A **sequential look-ahead procedure** to verify that
        precision remains stable in additional replications.

    Attributes
    ----------
    alpha : float
        Significance level for confidence interval calculations.
    half_width_precision : float
        Target CI half-width precision (i.e. percentage deviation of the
        confidence interval from the mean).
    initial_replications : int
        Number of replications to run before evaluating precision.
    look_ahead : int
        Number of additional replications to simulate for stability checks
        (adjusted proportionally when `n > 100`).
    replication_budget : int
        Maximum number of replications allowed.
    verbose : bool
        If True, prints the current replication count during execution.
    observer_factory : callable or None
        Callable returning a new observer instance for each metric. Should
        be a function, lambda, or class constructor taking no arguments.
        Returned object must follow the `AlgorithmObserver` protocol.
        If None, uses `ReplicationTabulizer`.
    n : int
        Current replication count (updated during execution).
    _n_solution : int
        Solution replication count once convergence is met (or replication
        budget if not met).
    stats : OnlineStatistics or None
        Tracks running mean, variance, and confidence interval metrics.

    References
    ----------
    Hoad, K., Robinson, S., & Davies, R. (2010). Automated selection of the
    number of replications for a discrete-event simulation. *Journal of the
    Operational Research Society*, 61(11), 1632-1644.
    https://www.jstor.org/stable/40926090
    """

    def __init__(
        self,
        alpha: float | None = 0.05,
        half_width_precision: float | None = 0.1,
        initial_replications: int | None = 3,
        look_ahead: int | None = 5,
        replication_budget: float | None = 1000,
        verbose: bool | None = False,
        observer_factory: Callable[[], AlgorithmObserver] | None = None,
    ):
        """
        Initialise the replications algorithm

        Parameters
        ----------
        alpha: float, optional (default = 0.05)
            Significance level for confidence interval calculations
            (CI level = 100 * (1 - alpha) %).
        half_width_precision: float, optional (default = 0.1)
            Target CI half-width precision (i.e. percentage deviation of the
            confidence interval from the mean).
        initial_replications : int
            Number of replications to run before evaluating precision.
        look_ahead: int, optional (default = 5)
            Number of additional replications to simulate for stability checks.
            When the number of replications n <= 100 the value of look ahead
            is used. When n > 100 then look_ahead / 100 * max(n, 100) is used.
        replication_budget: int, optional (default = 1000)
            Maximum number of replications allowed; algorithm stops if not
            converged by then. Useful for larger models where replication
            runtime is a constraint.
        verbose: bool, optional (default = False)
            If True, prints replication count progress.
        observer_factory : callable or None, optional (default = None)
            Callable returning a new observer instance for each metric. Should
            be a function, lambda, or class constructor taking no arguments.
            Returned object must follow the `AlgorithmObserver` protocol.
            If None, uses `ReplicationTabulizer`.

        Raises
        ------
        ValueError
            If parameter values are invalid (see `valid_inputs()`).
        """
        self.alpha = alpha
        self.half_width_precision = half_width_precision
        self.initial_replications = initial_replications
        self.look_ahead = look_ahead
        self.replication_budget = replication_budget
        self.verbose = verbose

        # Initially set n to number of initial replications
        self.n = self.initial_replications

        self._n_solution = self.replication_budget

        if observer_factory is None:
            observer_factory = ReplicationTabulizer
        self.observer_factory = observer_factory

        self.stats = None

        # Check validity of provided parameters
        self.valid_inputs()

    def valid_inputs(self):
        """
        Checks validity of provided parameters.

        Ensures:
          - `initial_replications` and `look_ahead` are non-negative integers.
          - `half_width_precision` is > 0.
          - `replication_budget` is less than `initial_replications`.
          - `observer` is class with `.dev` and `.summary_frame()`.

        Raises
        ------
        ValueError
            If any conditions are not met.
        """
        for p in [self.initial_replications, self.look_ahead]:
            if not isinstance(p, int) or p < 0:
                raise ValueError(f"{p} must be a non-negative integer.")

        if self.half_width_precision <= 0:
            raise ValueError("half_width_precision must be greater than 0.")

        if self.replication_budget < self.initial_replications:
            raise ValueError(
                "replication_budget must be less than initial_replications."
            )

        if self.observer_factory is not None:
            # Must be a callable (class, function, or lambda)
            if not callable(self.observer_factory):
                raise TypeError(
                    "'observer_factory' must be callable (a class, function, ",
                    "or lambda).",
                )

            # Instantiate a temporary observer to inspect
            try:
                obs_instance = self.observer_factory()
            except Exception as e:
                raise TypeError(
                    f"Could not instantiate {self.observer_factory}: {e}"
                ) from e

            # Must adhere to the AlgorithmObserver protocol
            if not isinstance(obs_instance, AlgorithmObserver):
                raise TypeError(
                    "Observer factory must return an object implementing the "
                    "AlgorithmObserver protocol."
                )

    def _klimit(self) -> int:
        """
        Determine the number of additional replications to check after the
        desired confidence interval precision is first reached.

        The look-ahead count scales with the total number of replications:
        - If n ≤ 100, returns the fixed `look_ahead` value.
        - If n > 100, returns `look_ahead / 100 * max(n, 100)`, rounded down.

        Returns
        -------
        int
            Number of additional replications to check precision stability.
            Returned value is always rounded down to the nearest integer.
        """
        return int((self.look_ahead / 100) * max(self.n, 100))

    def find_position(self, lst: list[float]):
        """
        Find the first position where element is below deviation, and this is
        maintained through the lookahead period.

        This is used to correct the ReplicationsAlgorithm, which cannot return
        a solution below the initial_replications.

        Parameters
        ----------
        lst : list[float]
            List of deviations.

        Returns
        -------
        int or None
            Minimum replications required to meet and maintain precision,
            or None if not found.
        """
        # Check if the list is empty or if no value is below the threshold
        if not lst or all(x is None or x >= self.half_width_precision for x in lst):
            return None

        # Find the first non-None value in the list
        start_index = pd.Series(lst).first_valid_index()

        # Iterate through the list, stopping when at last point where we still
        # have enough elements to look ahead
        if start_index is not None:
            for i in range(start_index, len(lst) - self.look_ahead):
                # Create slice of list with current value + lookahead
                # Check if all fall below the desired deviation
                if all(
                    value < self.half_width_precision
                    for value in lst[i : i + self.look_ahead + 1]
                ):
                    # Add one, so it is the number of reps, as is zero-indexed
                    return i + 1
        return None

    def select(
        self, model: ReplicationsAlgorithmModelAdapter, metrics: list[str]
    ) -> dict[str, int]:
        """
        Executes the replication algorithm, determining the necessary number
        of replications to achieve and maintain the desired precision.

        The process:
          1. Runs `initial_replications` of the model.
          2. Updates running statistics and calculates CI precision.
          3. If precision met, tests stability via the look-ahead procedure.
          4. Stops when stable precision is achieved or budget is exhausted.

        Parameters
        ----------
        model : ReplicationsAlgorithmModelAdapter
            Simulation model implementing `single_run(replication_index)`.
        metrics: list[str]
            The metrics to assess.

        Returns
        -------
        nreps : dict[str, int or None]
            Minimum replications required for each metric, or None if not
            achieved.
        summary_frame : pandas.DataFrame
            Table summarising deviation and CI metrics for all replications.

        Raises
        ------
        ValueError
            If the provided `model` is not an instance of
            `ReplicationsAlgorithmModelAdapter`.

        Warns
        -----
        UserWarning
            If convergence is not reached within the allowed replication
            budget.
        """
        # Check validity of provided model
        if not isinstance(model, ReplicationsAlgorithmModelAdapter):
            raise ValueError(ALG_INTERFACE_ERROR)

        # Create instances of observer for each metric
        observers = {metric: self.observer_factory() for metric in metrics}

        # Create tracking dictionary
        solutions = {
            metric: {
                "nreps": None,  # The solution
                "target_met": 0,  # Consecutive times target met
                "solved": False,  # Precision maintained over lookahead?
            }
            for metric in metrics
        }

        # If there are no initial replications, create empty instances of
        # OnlineStatistics for each metric
        if self.initial_replications == 0:
            stats = {
                metric: OnlineStatistics(
                    data=None, alpha=self.alpha, observer=observers[metric]
                )
                for metric in metrics
            }
        # If there are, run replications then create instances of
        # OnlineStatistics pre-loaded with data from initial replications
        else:
            initial_results = [
                model.single_run(rep) for rep in range(self.initial_replications)
            ]
            stats = {}
            for metric in metrics:
                stats[metric] = OnlineStatistics(
                    data=np.array([res[metric] for res in initial_results]),
                    alpha=self.alpha,
                    observer=observers[metric],
                )

        # Check which metrics meet precision after initial replications
        for metric in metrics:
            if stats[metric].deviation <= self.half_width_precision:
                solutions[metric]["nreps"] = self.n
                solutions[metric]["target_met"] = 1
                # If there is no lookahead, mark as solved
                if self._klimit() == 0:
                    solutions[metric]["solved"] = True

        while (
            sum(1 for v in solutions.values() if v["solved"]) < len(metrics)
            and self.n < self.replication_budget + self._klimit()
        ):
            new_result = model.single_run(self.n)
            self.n += 1
            for metric in metrics:
                # Only process metrics that have not yet stably met precision
                if not solutions[metric]["solved"]:
                    # Update running statistics with latest replication result
                    stats[metric].update(new_result[metric])

                    # Check if current deviation is within target precision
                    if stats[metric].deviation <= self.half_width_precision:
                        # Record solution, if not met in the last run
                        if solutions[metric]["target_met"] == 0:
                            solutions[metric]["nreps"] = self.n

                        # Increment consecutive precision counter
                        solutions[metric]["target_met"] += 1

                        # Mark as solved if precision has held for look-ahead
                        if solutions[metric]["target_met"] > self._klimit():
                            solutions[metric]["solved"] = True

                    else:
                        # Precision lost — reset counters / solution point
                        solutions[metric]["target_met"] = 0
                        solutions[metric]["nreps"] = None

        # Correction to result, replacing if stable solution was achieved
        # within initial replications
        for metric, dictionary in solutions.items():
            adj_nreps = self.find_position(observers[metric].dev)
            if adj_nreps is not None and dictionary["nreps"] is not None:
                if adj_nreps < dictionary["nreps"]:
                    solutions[metric]["nreps"] = adj_nreps

        # Extract minimum replications for each metric
        nreps = {metric: value["nreps"] for metric, value in solutions.items()}
        if None in nreps.values():
            unsolved = [k for k, v in nreps.items() if v is None]
            warnings.warn(f"WARNING: precision not reached for: {unsolved}")

        # Combine summary frames
        summary_frame = pd.concat(
            [
                observer.summary_table().assign(metric=metric)
                for metric, observer in observers.items()
            ]
        ).reset_index(drop=True)

        return nreps, summary_frame
