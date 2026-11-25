"""
Classes and functions to support time dependent samplingm in DES models.
"""

import itertools

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.random import SeedSequence

from sim_tools.distributions import DistributionRegistry

from typing import Optional, Tuple
import plotly.graph_objects as go
import plotly.io as pio


@DistributionRegistry.register()
class NSPPThinning:
    """
    Non Stationary Poisson Process via Thinning.

    Thinning is an acceptance-rejection approach to sampling
    inter-arrival times (IAT) from a time-dependent distribution
    where each time period follows its own exponential distribution.

    This implementation takes mean inter-arrival times as inputs, making it
    consistent with NumPy's exponential distribution parameterization.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        interval_width: float | None = None,
        random_seed1: int | SeedSequence | None = None,
        random_seed2: int | SeedSequence | None = None,
    ):
        """
        Non Stationary Poisson Process via Thinning.

        Time dependency is handled for a single table
        consisting of equally spaced intervals.

        Parameters
        ----------
        data: pandas.DataFrame
            DataFrame with time points and mean inter-arrival times.
            Columns should be "t" and "mean_iat" respectively.

        interval_width: float, optional (default=None)
            The width of each time interval. If None, it will be calculated
            from consecutive time points in the data. Required if data has only
            one row.

        random_seed1: int | SeedSequence, optional (default=None)
            Random seed for the exponential distribution

        random_seed2: int | SeedSequence, optional (default=None)
            Random seed for the uniform distribution used
            for acceptance/rejection sampling.
        """

        # -- Defensive parameter check added by TM as part of fix v1.0.4 --

        # data must be a dataframe
        if not isinstance(data, pd.DataFrame):
            raise TypeError(
                f"Data should be a DataFrame, but got {type(data).__name__}"
            )

        # Check required columns t and mean_iat are provided.
        required_cols = ["t", "mean_iat"]
        missing_cols = [col for col in required_cols if col not in data.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        # empty dataframe or only a single row.
        if data.empty:
            raise ValueError("Dataframe does not contain any rows. Add at least 2.")
        if len(data) == 1:
            msg = (
                "Dataframe contains a single time point. Add more of use "
                + "Exponential class."
            )
            raise ValueError(msg)

        # -- end of defensive code --

        self.data = data
        self.arr_rng = np.random.default_rng(random_seed1)
        self.thinning_rng = np.random.default_rng(random_seed2)

        # drop to numpy for lookup speed
        self.mean_iats = data["mean_iat"].to_numpy(dtype=float)

        # Get the minimum mean IAT (corresponds to the maximum arrival rate)
        self.min_iat = self.mean_iats.min()

        if self.min_iat <= 0:
            raise ValueError("Mean inter-arrival times must be positive")

        # Use provided interval width or calculate from data
        if interval_width is not None:
            self.interval = interval_width
        elif len(data) > 1:
            # Calculate from data (assumes all intervals are equal in length)
            self.interval = data.iloc[1]["t"] - data.iloc[0]["t"]
        else:
            raise ValueError(
                "With only one data point, interval_width must be provided"
            )

        # Pre-compute the acceptance probabilities. Added by TM v1.0.4
        self.accept_prob = self.min_iat / self.mean_iats
        self.n_periods = self.mean_iats.size

        self.rejects_last_sample = None

    def __repr__(self):
        """Return a string representation of the NSPPThinning instance."""
        # Truncate the data representation if too long
        max_len = 100
        data_str = repr(self.data)
        if len(data_str) > max_len:
            data_str = data_str[:max_len] + "..."

        # Return class name with both data and interval information
        return (
            f"{self.__class__.__name__}(data={data_str}, "
            + f"interval={self.interval})"
        )

    def sample(self, simulation_time: float) -> float:
        """
        Run a single iteration of acceptance-rejection
        thinning alg to sample the next inter-arrival time

        Parameters
        ----------
        simulation_time: float
            The current simulation time.

        Returns
        -------
        float
            The inter-arrival time
        """

        interarrival_time = 0.0

        for self.rejects_last_sample in itertools.count(start=0):
            # sample the next inter-arrival time and calculate clock time
            interarrival_time += self.arr_rng.exponential(self.min_iat)
            arrival_sim_clock_time = simulation_time + interarrival_time

            # get the t index of the candidate arrival
            # fix v1.0.4 to use 'candidate_arrival_time' Fix by Sammi Rosser
            t_idx = int(arrival_sim_clock_time // self.interval) % self.n_periods

            # accept or reject? TM modified v1.0.4 to use pre-computed probabilities
            u = self.thinning_rng.uniform()
            if u <= self.accept_prob[t_idx]:
                # accepted so return inter-arrival time
                return interarrival_time

        raise RuntimeError("Unreachable: thinning loop exited without accepting")


def nspp_simulation(
    arrival_profile: pd.DataFrame,
    run_length: float | None = None,
    n_reps: int | None = 1000,
) -> pd.DataFrame:
    """
    Generate a pandas dataframe that contains multiple replications of
    a non-stationary poisson process for the set arrival profile.

    This uses the sim-tools NSPPThinning class.

    Useful for validating the the NSPP has been set up correctly and is
    producing the desired profile for the simulation model.

    On each replication the function counts the number of arrivals during the
    intervals from the arrival profile.  Returns a data frame with reps (rows)
    and interval arrivals (columns).

    Parameters
    ----------
    arrival_profile: pandas.DataFrame
        The arrival profile is a pandas data frame containing 't',
        'arrival_rate' and 'mean_iat' columns.

    run_length: float, optional (default=None)
        How long should the simulation be run. If none then uses the last
        value in 't' + the interval (assumes equal width intervals)

    n_reps: int, optional (default=1000)
        The number of replications to run.

    Returns
    -------
    pd.DataFrame.


    """
    # replication results
    replication_results = []

    # multiple replications
    for rep in range(n_reps):
        # method for producing n non-overlapping streams
        seed_sequence = np.random.SeedSequence(rep)

        # Generate n high quality child seeds
        seeds = seed_sequence.spawn(2)

        # create nspp
        nspp_rng = NSPPThinning(
            data=arrival_profile, random_seed1=seeds[0], random_seed2=seeds[1]
        )

        # if no run length has been set....
        if run_length is None:
            run_length = (
                arrival_profile["t"].iloc[len(arrival_profile) - 1] + nspp_rng.interval
            )

        # list - each item is an interval in the arrival profile
        interval_samples = [0] * arrival_profile.shape[0]
        simulation_time = 0.0
        while simulation_time < run_length:
            iat = nspp_rng.sample(simulation_time)
            simulation_time += iat

            if simulation_time < run_length:
                # data collection: add one to count for hour of the day
                # note list NSPPThinning this assume equal intervals
                interval_of_day = int(simulation_time // nspp_rng.interval) % len(
                    arrival_profile
                )
                interval_samples[interval_of_day] += 1

        replication_results.append(interval_samples)

    # produce summary chart of arrivals per interval
    # format in a dataframe
    df_replications = pd.DataFrame(replication_results)
    df_replications.index = np.arange(1, len(df_replications) + 1)
    df_replications.index.name = "rep"

    return df_replications


# def nspp_plot(
#     arrival_profile: pd.DataFrame,
#     run_length: float | None = None,
#     n_reps: int | None = 1000,
# ) -> tuple[plt.Figure, plt.Axes]:
#     """
#     Generate a matplotlib chart to visualise a non-stationary poisson process
#     for the set arrival profile.

#     This uses the sim-tools NSPPThinning class.

#     Useful for validating the the NSPP has been set up correctly and is
#     producing the desired profile for the simulation model.

#     Parameters
#     ----------
#     arrival_profile: pandas.DataFrame
#         The arrival profile is a pandas data frame containing 't',
#         'arrival_rate' and 'mean_iat' columns.

#     run_length: float, optional (default=None)
#         How long should the simulation be run. If none then uses the last value
#         in 't' + the interval (assumes equal width intervals)

#     n_reps: int, optional (default=1000)
#         The number of replications to run.
#     """

#     # verification of arrival_profile

#     # is it a dataframe
#     if not isinstance(arrival_profile, pd.DataFrame):
#         raise ValueError(
#             "arrival_profile expected pd.DataFrame " + f"got {type(arrival_profile)}"
#         )

#     # all columns are present
#     required_columns = ["t", "arrival_rate", "mean_iat"]
#     for col in required_columns:
#         if col not in arrival_profile.columns:
#             raise ValueError(
#                 f"arrival_profile must contain "
#                 f"the following columns: {required_columns}. "
#             )

#     # generate the sample data
#     df_interval_results = nspp_simulation(arrival_profile, run_length, n_reps)

#     interval_means = df_interval_results.mean(axis=0)
#     interval_sd = df_interval_results.std(axis=0)

#     upper = interval_means + interval_sd
#     lower = interval_means - interval_sd
#     lower[lower < 0] = 0

#     # visualise
#     fig = plt.figure(figsize=(12, 3))
#     ax = fig.add_subplot()

#     # plot in this case returns a 2D line plot object
#     _ = ax.plot(arrival_profile["t"], interval_means, label="Mean")
#     _ = ax.fill_between(arrival_profile["t"], lower, upper, alpha=0.2, label="+-1SD")

#     # chart appearance
#     _ = ax.legend(loc="best", ncol=3)
#     _ = ax.set_ylim(
#         0,
#     )
#     _ = ax.set_xlim(0, arrival_profile.shape[0] - 1)
#     _ = ax.set_ylabel("arrivals")
#     _ = ax.set_xlabel("interval (from profile)")
#     _ = plt.xticks(arrival_profile["t"])

#     return fig, ax

@DistributionRegistry.register()
class NSPPDirect:
    """
    Non-Stationary Poisson Process via the Direct (Inversion) Algorithm.
    Used to sequentially generate inter-arrival times (IATs) of time
    dependent arrival processes.

    The rate profile is treated as repeating: arrivals are generated
    indefinitely, cycling back to the first segment once the end of the
    table is passed.

    In the direct algorithm we use K piecewise linear segments (straight
    lines) to represent cumulative expected arrivals over time. We have one
    "clock" that ticks in expected arrivals to the system. For a new arrival
    we walk up the y-axis to determine which segment to use, and convert the
    number of ticks into a simulation time by dividing the cumulative
    arrivals within that segment by the arrival rate of the segment. This
    gives the absolute time of the next arrival, Tn. The method sample() returns the
    IAT (Tn minus the previous arrival time), so the first call returns the
    time of the first arrival from t = 0.

    Algorithm:
    ----------

    The algorithm inverts the cumulative rate function Λ(t) to map
    unit-rate Poisson process event times (Sn) onto NSPP event times (Tn).

    With K piecewise-constant segments, the pre-computed tables are:
        b_k : right time-boundary of segment k  (b_0 = 0)
        r_k : constant arrival rate over segment k
        c_k : cumulative arrivals at right boundary of segment k  (c_0 = 0)

    Key inversion formula (Eq. 4, Harrod & Kelton 2006):
        Tn = b_{k-1} + (Sn - c_{k-1}) / r_k
    where k satisfies c_{k-1} < Sn <= c_k.

    Sn = arrival clock (cumulative expected arrivals, i.e. number of ticks)
    Tn = absolute simulation time of the nth arrival

    Parameters
    ----------
    data : pd.DataFrame
        Columns:
        "t"            - right boundary of each interval (b_k). Must be
                        finite, positive and strictly increasing.
        "arrival_rate" - piecewise-constant rate r_k for that segment
        "mean_iat"     - optional; 1 / arrival_rate. Accepted for interface
                        parity with NSPPThinning; not used internally.
    interval_width : float, optional
        Width of each time interval. Informational only (used in repr/plot,
        not by the inversion). Inferred from the first two rows when omitted.
    random_seed : int | SeedSequence, optional
        Seed for the U(0,1) RNG used to generate unit-rate Poisson IATs.

    Reference:
    ----------
    Harrod, S. & Kelton, W.D. (2006). Numerical Methods for Realizing
    Nonstationary Poisson Processes with Piecewise-Constant
    Instantaneous-Rate Functions. Simulation, 82(3), 147-157.
    DOI: 10.1177/0037549706065514
    """

    def __init__(
        self,
        data: pd.DataFrame,
        interval_width: Optional[float] = None,
        random_seed: Optional[int | SeedSequence] = None,
    ):
        self.data = data
        self.rng = np.random.default_rng(random_seed)

        # some basic validation of arrival rates
        if (data["arrival_rate"] < 0).any():
            raise ValueError("Arrival rates must be non-negative.")
        if (data["arrival_rate"] == 0).all():
            raise ValueError("At least one arrival rate must be positive.")

        if interval_width is not None:
            self.interval = float(interval_width)
        elif len(data) > 1:
            self.interval = float(data.iloc[1]["t"] - data.iloc[0]["t"])
        else:
            raise ValueError(
                "With only one data point, interval_width must be provided."
            )

        # Build r_k, b_k, c_k arrays.
        # Index 0 holds the origin (b_0=0, c_0=0); segments are 1-indexed.
        rates   = data["arrival_rate"].to_numpy(dtype=float)
        b_right = data["t"].to_numpy(dtype=float)
        b_left  = np.concatenate([[0.0], b_right[:-1]])
        widths  = b_right - b_left
        cumulative = np.cumsum(rates * widths)

        self._r = np.concatenate([[0.0], rates])       # r[0] unused
        self._b = np.concatenate([[0.0], b_right])     # b[0] = 0
        self._c = np.concatenate([[0.0], cumulative])  # c[0] = 0

        # Internal state
        self._Sn_prev: float = 0.0  # cumulative rate-1 time at last event
        self._Tn_prev: float = 0.0  # NSPP clock time at last event
        self._n = 0 # arrival count

    def __repr__(self) -> str:
        """
        String representation of the generator and its live state.

        Shows the profile's derived parameters (number of segments,
        cycle length, expected arrivals per cycle, rate range) plus
        the current generator state (simulation time, arrival clock,
        number of arrivals sampled), so that stateful objects can be
        distinguished at a glance in tracebacks and logs.
        """
        r = self._r[1:]
        s = (f"{self.__class__.__name__}(K={len(r)}, "
            f"cycle={self._b[-1]:g}, E[arr]/cycle={self._c[-1]:g}, "
            f"rates {r.min():g}-{r.max():g}")
        if (r == 0).any():
            s += f" ({int((r == 0).sum())} zero-rate)"
        s += f", at t={self._Tn_prev:.4g}, Sn={self._Sn_prev:.4g}, n={self._n})"
        return s

    def reset(self) -> None:
        """Reset internal state so the generator can be reused from t=0.
        Needed because the algorithm is stateful.  If the same object is used
        across replications then reset should be called."""
        self._Sn_prev = 0.0
        self._Tn_prev = 0.0
        self._n = 0

    def _invert(self, Sn_val: float) -> float:
        """
        Map a cumulative rate-1 value Sn onto an NSPP time Tn,
        supporting cyclic (repeating) rate profiles.

        Implements steps 5.4-5.5 of the direct algorithm pseudocode.
        """
        K = len(self._r) - 1
        C_total = self._c[-1]   # total arrivals per cycle
        T_cycle = self._b[-1]   # wall-clock length of one profile cycle

        # Handle cycles i.e. circling back to the first interval
        n_cycles  = int(Sn_val // C_total)
        Sn_rem    = Sn_val % C_total

        # Step 5.4: find segment k such that c[k-1] < Sn_rem <= c[k]
        # TM note: fine for small arrival profiles but maybe inefficient for large
        # is there a numpy way to do this quickly?
        k = 1
        while k < K and Sn_rem > self._c[k]:
            k += 1

        # Step 5.5: Eq. 4 inversion
        Tn_within = self._b[k - 1] + (Sn_rem - self._c[k - 1]) / self._r[k]
        return n_cycles * T_cycle + Tn_within

    def sample(self) -> float:
        """
        Generate the next inter-arrival time using the Direct Algorithm.

        The direct algorithm uses its own internal state
        (Sn, Tn) and does not require the caller to supply the 
        simulation clock time.

        Returns
        -------
        float
            Inter-arrival time until the next NSPP event.
        """
        # Steps 5.1–5.3: generate next unit-rate Poisson inter-event time
        u  = self.rng.uniform(0.0, 1.0)
        An = -np.log(u)
        Sn = self._Sn_prev + An

        # Steps 5.4–5.5: invert to absolute NSPP time, then derive IAT
        Tn  = self._invert(Sn)
        # inter-arrival time = current time - time of previous arrival
        iat = Tn - self._Tn_prev

        self._Sn_prev = Sn
        self._Tn_prev = Tn
        self._n += 1
        return iat
    
    def plot_cumulative_rate(
        self,
        arrivals=None,
        n_arrivals=0,
        random_seed=None,
        show_grid_lines=True,
        units_y=None,
        units_x=None
    ) -> go.Figure:
        """
        Interactive Plotly recreation of Figure 2 from Harrod & Kelton (2006).

        Plots the cumulative rate function Λ(t) with:
        - Piecewise-linear Λ(t) curve (T on x-axis, S on y-axis)
        - b_k boundaries marked on the time axis
        - c_k boundaries marked on the arrival-clock axis
        - r_k slope annotations for each segment
        - Optional: sampled arrivals (Tn, Sn) with dashed projection lines
            showing the inversion step T_n = Λ⁻¹(S_n)

        Parameters
        ----------
        arrivals : list of (Tn, Sn) tuples, optional
            Pre-computed arrival points to overlay. If None and n_arrivals > 0,
            arrivals are sampled internally using random_seed.
        n_arrivals : int, optional (default=0)
            Number of arrivals to simulate and overlay. Set to 0 to show
            the cumulative rate function only.
        random_seed : int, optional
            Seed for the internal RNG when sampling arrivals.
        show_grid_lines : bool, optional (default=True)
            Draw dashed projection lines from each arrival point to both axes,
            illustrating the inversion step.

        Returns
        -------
        plotly.graph_objects.Figure
        """

        # Build arrival rate Λ(t) curve 
        t_points = [0.0]
        c_points = [0.0]
        for k in range(1, len(self._b)):
            t_points.append(self._b[k])
            c_points.append(self._c[k])

        # Sample arrivals to display
        if arrivals is None and n_arrivals > 0:
            # safer to create internal NSPPDirect object 
            # in case user doesn't called reset on S_n and T_n
            rng_copy = NSPPDirect(
                data=self.data,
                interval_width=self.interval,
                random_seed=random_seed
            )

            Sn_running = 0.0
            arrivals = []
            for _ in range(n_arrivals):
                u  = rng_copy.rng.uniform(0.0, 1.0)
                An = -np.log(u)
                Sn = Sn_running + An
                Tn = rng_copy._invert(Sn)
                arrivals.append((Tn, Sn))
                Sn_running = Sn

        # Axis limits 
        t_max = self._b[-1] * 1.05
        c_max = self._c[-1] * 1.05

        fig = go.Figure()

        # 1. Plot Λ(t) cumulative rate curve
        fig.add_trace(go.Scatter(
            x=t_points, y=c_points,
            mode="lines",
            name="Λ(t) cumulative rate",
            line=dict(width=3),
            hovertemplate="t = %{x:.2f}<br>Λ(t) = %{y:.4f}<extra></extra>",
        ))

        # 2. b_k boundary markers (vertical dotted lines + labels)
        for k in range(1, len(self._b) - 1):
            fig.add_shape(type="line",
                x0=self._b[k], x1=self._b[k],
                y0=0, y1=self._c[k],
                line=dict(color="black", width=1, dash="dot"),
                layer="below",
            )
            fig.add_annotation(
                x=self._b[k], y=-c_max * 0.04,
                text=f"b<sub>{k}</sub>",
                showarrow=False, font=dict(size=11), xanchor="center",
            )

        # 3. c_k boundary markers (horizontal dotted lines + labels)
        for k in range(1, len(self._c) - 1):
            fig.add_shape(type="line",
                x0=0, x1=self._b[k],
                y0=self._c[k], y1=self._c[k],
                line=dict(color="black", width=1, dash="dot"),
                layer="below",
            )
            fig.add_annotation(
                x=-t_max * 0.03, y=self._c[k],
                text=f"c<sub>{k}</sub>",
                showarrow=False, font=dict(size=11), xanchor="right",
            )

        # 4. r_k slope labels at segment midpoints
        for k in range(1, len(self._r)):
            t_mid = (self._b[k-1] + self._b[k]) / 2
            c_mid = (self._c[k-1] + self._c[k]) / 2
            fig.add_annotation(
                x=t_mid, y=c_mid,
                text=f"r<sub>{k}</sub>",
                showarrow=False,
                font=dict(size=12, color="black"),
                bgcolor="rgba(255,255,255,0.7)",
                xanchor="center",
            )

        # 5. Arrival projection lines and points
        if arrivals:
            colors = pio.templates[pio.templates.default].layout.colorway
            arr_color = colors[1] if len(colors) > 1 else "red"

            for i, (Tn, Sn) in enumerate(arrivals):
                # if showing mapping from Λ(t) to arrival clock t
                if show_grid_lines:
                    # Vertical drop to time axis
                    fig.add_shape(type="line",
                        x0=Tn, x1=Tn, y0=0, y1=Sn,
                        line=dict(color=arr_color, width=1, dash="dash"),
                        layer="below",
                    )
                    # Horizontal projection to arrival-clock axis
                    fig.add_shape(type="line",
                        x0=0, x1=Tn, y0=Sn, y1=Sn,
                        line=dict(color=arr_color, width=1, dash="dash"),
                        layer="below",
                    )
                fig.add_trace(go.Scatter(
                    x=[Tn], y=[Sn],
                    mode="markers",
                    marker=dict(size=9, color=arr_color, symbol="circle"),
                    showlegend=False,
                    hovertemplate=(
                        f"<b>Arrival {i+1}</b><br>"
                        f"T<sub>n</sub> = {Tn:.4f}<br>"
                        f"S<sub>n</sub> = {Sn:.4f}"
                        "<extra></extra>"
                    ),
                ))

            # Single legend entry for all arrival points
            fig.add_trace(go.Scatter(
                x=[None], y=[None], mode="markers",
                marker=dict(size=9, color=arr_color),
                name="Arrival (Tₙ, Sₙ)",
            ))

        fig.update_layout(
            title=dict(text=(
                "Cumulative Arrival Function Λ(t)"
            #    "Dashed lines show inversion Tₙ = Λ⁻¹(Sₙ)"
            )),
            legend=dict(orientation="h", yanchor="bottom",
                        y=1.05, xanchor="center", x=0.5),
        )

        # Add units to axes labels

        y_label = "Cumulative Arrivals Λ(t)"
        x_label = "Time t"
        if units_y:
            y_label += f" ({units_y})"

        if units_x:
            x_label += f" ({units_x})"

        fig.update_xaxes(title_text=x_label, range=[-t_max * 0.05, t_max])
        fig.update_yaxes(title_text=y_label,  range=[-c_max * 0.06, c_max])
        return fig
        
def nspp_direct_simulation(
    arrival_profile: pd.DataFrame,
    run_length: Optional[float] = None,
    n_reps: Optional[int] = 1000,
) -> pd.DataFrame:
    """
    Generate a pandas DataFrame containing multiple replications of a
    non-stationary Poisson process using the Direct (Inversion) Algorithm.

    Equivalent to nspp_simulation() but uses NSPPDirect instead of
    NSPPThinning.  Useful for validating that NSPPDirect reproduces the
    desired arrival profile and for cross-checking against nspp_simulation.

    On each replication counts arrivals per interval from the profile.
    Returns a DataFrame with reps as rows and intervals as columns.

    Parameters
    ----------
    arrival_profile : pd.DataFrame
        Must contain columns 't', 'arrival_rate', and 'mean_iat'.
    run_length : float, optional (default=None)
        How long to run each replication. Defaults to the last value of
        't' in the profile (= one complete cycle).
    n_reps : int, optional (default=1000)
        Number of replications.

    Returns
    -------
    pd.DataFrame
        Shape (n_reps, n_intervals). Index labelled 'rep' (1-based).

    Notes
    -----
    run_length defaults to the last 't' value (one complete cycle) rather
    than last 't' + interval. Using last 't' + interval would add a partial
    second cycle and double-count the first interval's arrivals.
    """
    replication_results = []

    if run_length is None:
        run_length = float(arrival_profile["t"].iloc[-1])

    for rep in range(n_reps):
        seed_sequence = np.random.SeedSequence(rep)
        seeds = seed_sequence.spawn(1)
        nspp_rng = NSPPDirect(data=arrival_profile, random_seed=seeds[0])

        interval_samples = [0] * arrival_profile.shape[0]
        simulation_time = 0.0

        while simulation_time < run_length:
            iat = nspp_rng.sample()
            simulation_time += iat
            if simulation_time < run_length:
                interval_of_day = (
                    int(simulation_time // nspp_rng.interval)
                    % len(arrival_profile)
                )
                interval_samples[interval_of_day] += 1

        replication_results.append(interval_samples)

    df_replications = pd.DataFrame(replication_results)
    df_replications.index = np.arange(1, len(df_replications) + 1)
    df_replications.index.name = "rep"
    return df_replications


# def nspp_direct_plot(
#     arrival_profile: pd.DataFrame,
#     run_length: Optional[float] = None,
#     n_reps: Optional[int] = 1000,
# ) -> Tuple[plt.Figure, plt.Axes]:
#     """
#     Generate a matplotlib chart to visualise a non-stationary Poisson
#     process produced by the Direct (Inversion) Algorithm.

#     Equivalent to nspp_plot() but uses NSPPDirect. Plots simulated mean
#     arrivals per interval (±1 SD band) alongside the expected arrivals
#     (rate × interval_width) for direct visual validation.

#     Parameters
#     ----------
#     arrival_profile : pd.DataFrame
#         Must contain columns 't', 'arrival_rate', and 'mean_iat'.
#     run_length : float, optional (default=None)
#         Run length per replication. Defaults to last 't' (one full cycle).
#     n_reps : int, optional (default=1000)
#         Number of replications.

#     Returns
#     -------
#     fig : plt.Figure
#     ax  : plt.Axes
#     """
#     if not isinstance(arrival_profile, pd.DataFrame):
#         raise ValueError(
#             f"arrival_profile expected pd.DataFrame got {type(arrival_profile)}"
#         )
#     required_columns = ["t", "arrival_rate", "mean_iat"]
#     for col in required_columns:
#         if col not in arrival_profile.columns:
#             raise ValueError(
#                 f"arrival_profile must contain "
#                 f"the following columns: {required_columns}."
#             )

#     df_interval_results = nspp_direct_simulation(
#         arrival_profile, run_length, n_reps
#     )
#     interval_means = df_interval_results.mean(axis=0)
#     interval_sd    = df_interval_results.std(axis=0)
#     upper = interval_means + interval_sd
#     lower = (interval_means - interval_sd).clip(lower=0)

#     _tmp = NSPPDirect(data=arrival_profile)
#     expected = arrival_profile["arrival_rate"] * _tmp.interval

#     fig = plt.figure(figsize=(12, 4))
#     ax  = fig.add_subplot()
#     ax.plot(arrival_profile["t"], interval_means, label="Simulated mean")
#     ax.fill_between(
#         arrival_profile["t"], lower, upper, alpha=0.2, label="±1 SD"
#     )
#     ax.plot(
#         arrival_profile["t"], expected.values,
#         linestyle="--", color="black",
#         label="Expected (rate × interval)",
#     )
#     ax.legend(loc="best", ncol=3)
#     ax.set_ylim(0)
#     ax.set_xlim(0, arrival_profile.shape[0] - 1)
#     ax.set_ylabel("Arrivals per interval")
#     ax.set_xlabel("Interval right boundary (min)")
#     plt.xticks(arrival_profile["t"])
#     plt.tight_layout()
#     return fig, ax


def _nspp_interval_plot(
    df_results: pd.DataFrame,
    interval: float,
    algorithm_label: str,
    ax: plt.Axes,
    kind: str = "bar",
    offset: float = 0.0,
    n_series: int = 1,
    expected: np.ndarray | None = None,
    show_error_bars: bool = True
) -> tuple[np.ndarray, np.ndarray]:
    """
    Plot simulated arrivals per interval for ONE algorithm.

    Bins are positioned at interval midpoints and labelled by time range,
    so the encoding is independent of the 't' column convention
    (interval start for thinning data vs right boundary for direct data).

    Parameters
    ----------
    df_results : pd.DataFrame
        Replications (rows) x intervals (columns) counts, as returned by
        nspp_simulation() / nspp_direct_simulation().
    interval : float
        Width of each interval, in the profile's time units.
    algorithm_label : str
        Legend label for this series.
    ax : plt.Axes
        Axes to draw on.
    kind : str, optional (default="bar")
        "bar" suits profiles with few intervals; "line" suits long
        profiles (e.g. half-hourly over 24h) where bars become clutter.
    offset : float, optional (default=0.0)
        x-offset for grouped bars when plotting more than one series
        on the same axes.
    n_series : int, optional (default=1)
        Number of series sharing the axes; controls bar width.
    expected : np.ndarray, optional (default=None)
        Expected arrivals per interval (rate x interval). Drawn once;
        pass to one call only to avoid a duplicate legend entry.

    Returns
    -------
    (midpoints, means) as np.ndarray
    """
    n_bins = df_results.shape[1]
    mid = (np.arange(n_bins) + 0.5) * interval
    means = df_results.mean(axis=0).to_numpy()
    sds = df_results.std(axis=0).to_numpy()

    if kind == "bar":
        width = interval * 0.8 / n_series
        yerr = sds if show_error_bars else None
        ax.bar(mid + offset, means, width=width, yerr=yerr, capsize=2,
               label=f"{algorithm_label} (simulated mean)")
    elif kind == "line":
        ax.plot(mid + offset, means, marker="o",
                label=f"{algorithm_label} (simulated mean)")
        ax.fill_between(mid + offset, means - sds, means + sds, alpha=0.2)
    else:
        raise ValueError(f"kind must be 'bar' or 'line', got {kind!r}")

    if expected is not None:
        ax.plot(mid, expected, "kD--", ms=7, label="Expected (rate \u00d7 interval)")

    ax.set_xticks(mid)
    if kind == "bar":
        ax.set_xticklabels(
            [f"{int(k * interval)}\u2013{int((k + 1) * interval)}"
             for k in range(n_bins)]
        )
    else:
        ax.set_xlabel("time")
    ax.set_ylabel("arrivals per interval")
    ax.set_ylim(0, None)
    return mid, means


def nspp_plot(arrival_profile, run_length=None, n_reps=1000, kind="bar", include_expected=True, show_error_bars=True):
    df = nspp_simulation(arrival_profile, run_length, n_reps)
    interval = float(arrival_profile.iloc[1]["t"] - arrival_profile.iloc[0]["t"])
    expected = None
    if include_expected:
        expected = arrival_profile["arrival_rate"].to_numpy() * interval
    fig, ax = plt.subplots(figsize=(12, 4))
    _nspp_interval_plot(df, interval, "Thinning", ax, kind=kind, expected=expected, show_error_bars=show_error_bars)
    ax.set_title("NSPP sense-check (thinning)")
    ax.legend()
    return fig, ax

def nspp_direct_plot(arrival_profile, run_length=None, n_reps=1000, kind="bar", include_expected=True, show_error_bars=True):
    df = nspp_direct_simulation(arrival_profile, run_length, n_reps)
    interval = float(arrival_profile.iloc[1]["t"] - arrival_profile.iloc[0]["t"])
    expected = None
    if include_expected:
        expected = arrival_profile["arrival_rate"].to_numpy() * interval
    fig, ax = plt.subplots(figsize=(12, 4))
    _nspp_interval_plot(df, interval, "Direct", ax, kind=kind, expected=expected, show_error_bars=show_error_bars)
    ax.set_title("NSPP sense-check (direct)")
    ax.legend()
    return fig, ax