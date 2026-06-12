"""
Statistical distribution classes for simulation modeling.

This module provides a collection of statistical distribution classes designed
for simulation applications. Each distribution implements the Distribution
protocol, which requires a sample() method that generates random values
according to the distribution's parameters.

Features
--------
- Consistent interface via the Distribution protocol
- Independent random number streams for each distribution instance
- Support for reproducible sampling via random seeds
- Implementation of distributions not directly available in scipy or numpy
- Registry system for dynamic creation and configuration of distributions

Distributions
------------
The module includes common statistical distributions such as:
- Exponential, Normal, Lognormal, Uniform, Triangular, Beta
- Gamma, Weibull, Erlang, ErlangK, Poisson
- Bernoulli, Discrete, PearsonV, PearsonVI
- Empirical distributions (ContinuousEmpirical, RawEmpirical)
- Utility distributions (FixedDistribution, CombinationDistribution,
  TruncatedDistribution)

Random Number Generation
-----------------------
Each distribution manages its own random number generator instance. All
distributions that accept a random_seed parameter support:
- Integer seeds for basic reproducibility
- numpy.random.SeedSequence objects for advanced stream management
- None for auto-generated seeds

Distribution Registry
--------------------
The DistributionRegistry provides centralized management for all distribution
classes:
- Register custom distribution classes with the
  @DistributionRegistry.register() decorator
- Create distribution instances by name with DistributionRegistry.create()
- Generate multiple distributions with statistically independent seeds using
  create_batch()
- Create configuration templates with get_template() for streamlined setup
- Support for both dictionary and list-based batch configurations

Examples
--------
Basic usage:
>>> from simtools.distributions import Normal
>>> norm_dist = Normal(mean=10, sigma=2)
>>> norm_dist.sample()  # Single sample
10.436523
>>> norm_dist.sample(3)  # Multiple samples
array([10.02, 12.21, 9.33])

Using SeedSequence for multiple streams:
>>> import numpy as np
>>> from simtools.distributions import Exponential, Uniform
>>> seed_seq = np.random.SeedSequence(12345)
>>> seeds = seed_seq.spawn(2)
>>> exp_dist = Exponential(mean=5, random_seed=seeds[0])
>>> uni_dist = Uniform(low=0, high=10, random_seed=seeds[1])

Using the DistributionRegistry:
>>> from simtools.distributions import DistributionRegistry
>>> # Create a single distribution
>>> exp_dist = DistributionRegistry.create("Exponential", mean=5.0)
>>>
>>> # Create multiple distributions with independent seeds
>>> config = {
...     "arrivals": {"class_name": "Exponential", "params": {"mean": 5.0}},
...     "service_times": {"class_name": "Normal", "params": {"mean": 10.0,
                                                             "sigma": 2.0}}
... }
>>> dists = DistributionRegistry.create_batch(config, main_seed=12345)
>>> arrivals = dists["arrivals"]
>>> service_times = dists["service_times"]

Notes
-----
All distribution parameters follow the conventions described in "Simulation
Modeling and Analysis" (Law, 2007) where applicable.
"""

import inspect
import json
import math
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Protocol,
    Tuple,
    TypeVar,
    Union,
    runtime_checkable,
)

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from numpy.random import SeedSequence
from numpy.typing import ArrayLike, NDArray

from sim_tools._validation import (
    is_integer,
    is_non_negative,  # >= 0 e.g. for location
    is_numeric,
    is_ordered_pair,
    is_ordered_triplet,
    is_positive,  # > 0
    is_positive_array,
    is_probability,
    is_probability_vector,
    validate,
)

T = TypeVar("T", bound=type)


@runtime_checkable
class Distribution(Protocol):
    """
    Distribution protocol defining the interface for probability distributions.

    Any class implementing this protocol should provide a sampling mechanism
    that generates random values according to a specific probability
    distribution.
    """

    def sample(
        self, size: Optional[Union[int, Tuple[int, ...]]] = None
    ) -> Union[float, NDArray[np.float64]]:
        """
        Generate random samples from the distribution.

        Parameters
        ----------
        size : Optional[Union[int, Tuple[int, ...]]], default=None
            The number/shape of samples to generate:
            - If None: returns a single sample as a float
            - If int: returns a 1-D array with that many samples
            - If tuple of ints: returns an array with that shape

        Returns
        -------
        Union[float, NDArray[np.float64]]
            Random samples from the distribution:
            - A single float when size is None
            - A numpy array of floats with shape determined by size parameter

        Examples
        --------
        >>> dist = SomeDistribution(params)
        >>> single_sample = dist.sample()  # Returns a float
        >>> array_1d = dist.sample(10)  # Returns 1D array with 10 samples
        >>> array_2d = dist.sample((2, 3))  # Returns 2×3 array of samples
        """


def spawn_seeds(n_streams: int, main_seed: Optional[int] = None):
    """
    Generate multiple statistically independent random seeds.

    This function creates a set of SeedSequence objects that are guaranteed
    to produce independent streams of random numbers. This is crucial for
    ensuring that multiple random number generators don't produce correlated
    outputs, which could bias simulation results.

    Parameters
    ----------
    n_streams : int
        The number of independent seed sequences to generate.
        Must be a positive integer.

    main_seed : Optional[int], default=None
        Master seed that determines all generated sequences.
        If None, a random entropy source is used, making results
        non-reproducible across runs. Providing a value enables
        reproducible sequences.

    Returns
    -------
    List[np.random.SeedSequence]
        A list of n_streams SeedSequence objects that can be used
        to initialize random number generators with independent streams.

    Notes
    -----
    This approach is preferred over manually creating seeds because
    it uses NumPy's entropy pool management to guarantee statistical
    independence between streams, avoiding subtle correlations that
    might occur with manually chosen seeds.

    Examples
    --------
    >>> seeds = spawn_seeds(3, main_seed=12345)
    >>> rng1 = np.random.default_rng(seeds[0])
    >>> rng2 = np.random.default_rng(seeds[1])
    >>> rng3 = np.random.default_rng(seeds[2])
    """
    seed_sequence = np.random.SeedSequence(main_seed)
    seeds = seed_sequence.spawn(n_streams)
    return seeds


class DistributionRegistry:
    """
    Registry for probability distribution classes with batch creation
    capabilities.

    The DistributionRegistry provides a central repository for registering
    distribution classes and instantiating them from configuration data. This
    facilitates dynamic creation of distribution objects based on runtime
    configuration, supporting scenarios like simulation models, statistical
    analysis, or any application requiring configurable random distributions.

    Key features:
    - Register distribution classes with a simple decorator
    - Create distribution instances from class names and parameters
    - Batch create multiple distributions from a dictionary or list
    - Automatic generation of statistically independent random seeds

    Examples
    --------
    Register distribution classes:

    >>> @DistributionRegistry.register()
    ... class Exponential:
    ...     def __init__(self, mean, random_seed=None):
    ...         self.rng = np.random.default_rng(random_seed)
    ...         self.mean = mean
    ...
    ...     def sample(self, size=None):
    ...         return self.rng.exponential(self.mean, size=size)
    ...
    >>> @DistributionRegistry.register("uniform_dist")  # Custom name
    ... class Uniform:
    ...     def __init__(self, low, high, random_seed=None):
    ...         self.rng = np.random.default_rng(random_seed)
    ...         self.low = low
    ...         self.high = high
    ...
    ...     def sample(self, size=None):
    ...         return self.rng.uniform(self.low, self.high, size=size)

    Create a distribution with parameters:

    >>> exp_dist = DistributionRegistry.create("Exponential", mean=2.0)
    >>> exp_dist.sample(5)  # Generate 5 samples

    Create multiple distributions from a flat configuration:

    >>> config = {
    ...     "arrivals": {
    ...         "class_name": "Exponential",
    ...         "params": {"mean": 5.0}
    ...     },
    ...     "service_times": {
    ...         "class_name": "uniform_dist",
    ...         "params": {"low": 1.0, "high": 3.0}
    ...     }
    ... }
    >>> distributions = DistributionRegistry.create_batch(config,
    ...                                                    main_seed=12345)
    >>> arrivals = distributions["arrivals"]
    >>> service_times = distributions["service_times"]

    Create distributions from a nested configuration:

    >>> config = {
    ...     "call": {
    ...         "C1": {"class_name": "Exponential", "params": {"mean": 10.0}},
    ...         "C2": {"class_name": "Exponential", "params": {"mean": 8.0}},
    ...     },
    ...     "response_time": {
    ...         "C1": {
    ...             "class_name": "Lognormal",
    ...             "params": {"mean": 30.0, "stdev": 5.0},
    ...         },
    ...     },
    ... }
    >>> dists = DistributionRegistry.create_batch(
    ...     config, main_seed=42, preserve_structure=True
    ... )
    >>> dists["call"]["C1"].sample(5)
    >>> dists["response_time"]["C1"].sample(5)

    Notes
    -----
    When creating distributions in batch with a main_seed, each distribution
    receives its own statistically independent seed derived from the main seed.
    This ensures proper statistical independence between random number streams
    while maintaining overall reproducibility through the main seed.

    When `preserve_structure=True`, the output mirrors the shape of the
    input configuration, so nested configs produce nested dicts of instances.
    When `preserve_structure=False` (default), all leaf distributions are
    flattened into a single dict whose keys are the path components joined by
    underscores (e.g. `"call_C1"`).

    With `sort=True` (default), keys at every nesting level are sorted
    alphabetically before seeds are assigned. This ensures that seed
    assignment is stable even if the insertion order of keys in the config
    dict changes between runs.
    """

    _registry: Dict[str, type] = {}

    @classmethod
    def register(cls, name: Optional[str] = None) -> Callable:
        """
        Decorator to register a distribution class in the registry.

        Parameters
        ----------
        name : Optional[str], default=None
            Name to register the class under. If None, uses the class name.

        Returns
        -------
        Callable
            Decorator function that registers the class
        """

        def decorator(distribution_class: T) -> T:
            nonlocal name
            if name is None:
                name = distribution_class.__name__
            cls._registry[name] = distribution_class
            return distribution_class

        return decorator

    @classmethod
    def get(cls, name: str) -> type:
        """
        Get a distribution class by name.

        Parameters
        ----------
        name : str
            Name of the registered distribution class

        Returns
        -------
        type
            The registered distribution class

        Raises
        ------
        ValueError
            If the distribution name is not found in the registry
        """
        if name not in cls._registry:
            raise ValueError(f"Distribution '{name}' not found in registry")
        return cls._registry[name]

    @classmethod
    def create(cls, name: str, **params):
        """
        Create a distribution instance by name.

        Parameters
        ----------
        name : str
            Name of the registered distribution class
        **params
            Parameters to pass to the distribution constructor

        Returns
        -------
        Any
            Instance of the requested distribution class
        """
        distribution_class = cls.get(name)
        return distribution_class(**params)

    @classmethod
    def _is_dist_config(cls, obj) -> bool:
        """
        Return True if `obj` is a valid distribution config leaf.

        A valid leaf is a dict with exactly the keys `'class_name'` and
        `'params'` and no others.

        Parameters
        ----------
        obj : object
            The object to test.

        Returns
        -------
        bool
        """
        return isinstance(obj, dict) and set(obj.keys()) == {"class_name", "params"}

    @classmethod
    def create_batch(
        cls,
        config: Union[List[Dict], Dict[str, Dict]],
        main_seed: Optional[int] = None,
        sort: Optional[bool] = True,
        preserve_structure: bool = False,
    ) -> Union[List, Dict]:
        """
        Create multiple distributions from a configuration dictionary or list.

        Accepts both flat and arbitrarily nested dict configurations. Every
        leaf must be a dict with exactly the keys `'class_name'` and
        `'params'`.

        Parameters
        ----------
        config : Union[List[Dict], Dict[str, Dict]]
            One of:

            - A **list** of distribution configs, each with `'class_name'`
              and `'params'`. Returns a list of instances.
            - A **flat dict** mapping names to distribution configs:

                  {
                      "arrivals": {
                          "class_name": "Exponential",
                          "params": {"mean": 5.0},
                      }
                  }

            - A **nested dict** where intermediate keys group distributions
              and leaves are distribution configs:

                  {
                      "call": {
                          "C1": {
                              "class_name": "Exponential",
                              "params": {"mean": 10.0},
                          }
                      }
                  }

        main_seed : Optional[int], default=None
            Master seed to generate individual seeds for each distribution.
            If None, random seeds will still be generated for independence.
        sort : Optional[bool], default=True
            If True, keys at every nesting level are sorted alphabetically
            before seeds are assigned. This ensures deterministic seed
            assignment even if config key insertion order changes between
            runs. Not relevant for top-level lists.
        preserve_structure : Optional[bool], default=False
            Controls the shape of the returned dict when `config` is a dict.

            - `False` (default): all distributions are returned in a single
              flat dict whose keys are the path components of each leaf joined
              by underscores, e.g. `"call_C1"`.
            - `True`: the returned dict mirrors the nesting of the input
              config, so `result["call"]["C1"]` gives the distribution
              directly.

            Has no effect when `config` is a list.

        Returns
        -------
        Union[List, Dict]
            - A list of distribution instances if `config` was a list.
            - A flat dict of distribution instances if `config` was a dict
              and `preserve_structure=False`.
            - A nested dict of distribution instances if `config` was a dict
              and `preserve_structure=True`.

        Raises
        ------
        TypeError
            If `config` is neither a list nor a dictionary.
        ValueError
            If any distribution config is malformed, if a required key is
            missing or an unexpected key is present, or if the config
            contains values that are neither dicts, lists, nor valid
            distribution configs.
        """
        flat_items = []

        def walk(node, path=()):
            if cls._is_dist_config(node):
                flat_items.append((path, node))
                return

            if isinstance(node, dict):
                # Detect a malformed distribution config: has one of the
                # expected keys but not the exact right set, which most
                # likely means a typo or missing key in a leaf config.
                if "class_name" in node or "params" in node:
                    expected = {"class_name", "params"}
                    raise ValueError(
                        f"Distribution config at path {path!r} must have "
                        f"ONLY the keys {expected}. "
                        f"Found keys: {set(node.keys())}"
                    )
                items = node.items()
                if sort:
                    items = sorted(items, key=lambda kv: kv[0])
                for key, value in items:
                    walk(value, path + (key,))
                return

            if isinstance(node, list):
                for i, value in enumerate(node):
                    walk(value, path + (i,))
                return

            raise ValueError(
                f"Expected a distribution config dict, a nested grouping "
                f"dict, or a list at path {path!r}. "
                f"Got {type(node).__name__!r}."
            )

        if isinstance(config, list):
            seeds = spawn_seeds(len(config), main_seed)
            return [
                cls._validate_and_create(dist_config, seeds[i])
                for i, dist_config in enumerate(config)
            ]

        if not isinstance(config, dict):
            raise TypeError(
                "Configuration must be a list or dictionary, "
                f"got {type(config).__name__!r}."
            )

        walk(config)
        seeds = spawn_seeds(len(flat_items), main_seed)

        created = [
            (path, cls._validate_and_create(dist_config, seeds[i]))
            for i, (path, dist_config) in enumerate(flat_items)
        ]

        if not preserve_structure:
            return {"_".join(map(str, path)): obj for path, obj in created}

        result: Dict = {}
        for path, obj in created:
            cursor = result
            for part in path[:-1]:
                cursor = cursor.setdefault(part, {})
            cursor[path[-1]] = obj
        return result

    @classmethod
    def _validate_and_create(cls, dist_config, seed):
        """
        Validate a distribution config, inject a random seed, and instantiate.

        Checks that `dist_config` has exactly the keys `'class_name'` and
        `'params'`, injects `random_seed` into the params, then creates
        and returns the distribution instance.

        Parameters
        ----------
        dist_config : dict
            Dictionary specifying the distribution configuration. Must have
            keys `'class_name'` (str) and `'params'` (dict), and no others.
        seed : int
            The seed to include in the distribution's parameters (as
            `'random_seed'`).

        Returns
        -------
        instance
            The created distribution instance.

        Raises
        ------
        ValueError
            If `dist_config` is not a dict, or does not have exactly the
            expected keys.
        """
        # Check config is a dictionary
        if not isinstance(dist_config, dict):
            raise ValueError("Each distribution config must be a dict.")

        # Require exactly 'class_name' and 'params' as keys.
        expected_keys = {"class_name", "params"}
        keys = set(dist_config.keys())
        if keys != expected_keys:
            raise ValueError(
                "Distribution config must have ONLY the keys "
                f"{expected_keys}. Found keys: {keys}"
            )

        # Copy params and inject the random seed.
        params = dist_config["params"].copy()

        # Only inject random seed if class constructor accepts it
        distribution_class = cls.get(dist_config["class_name"])
        sig = inspect.signature(distribution_class.__init__)
        if "random_seed" in sig.parameters:
            params["random_seed"] = seed

        # Instantiate and return the distribution object.
        return cls.create(dist_config["class_name"], **params)

    @classmethod
    def get_template(cls, format: str = "json", indent: int = 2) -> Union[Dict, str]:
        """
        Generate a template configuration containing all registered
        distributions.

        This helper method creates a template that includes all registered
        distribution types with appropriate dummy parameters. Users can modify
        this template and pass it directly to `create_batch()` to instantiate
        their distributions.

        Parameters
        ----------
        format : str, default="json"
            Output format: `'dict'` for a Python dictionary or `'json'`
            for a JSON string.
        indent : int, default=2
            Indentation for JSON formatting (only used when
            `format='json'`).

        Returns
        -------
        Union[Dict, str]
            Either a dictionary (if`format='dict'`) or a JSON string (if
            `format='json'`) containing template configurations for all
            registered distributions.

        Examples
        --------
        >>> template = DistributionRegistry.get_template(format='dict')
        >>> print(list(template.keys()))
        ['Exponential_example', 'Normal_example', 'Uniform_example', ...]

        >>> template = DistributionRegistry.get_template(format='json')
        >>> print(template[:70])
        {
          "Exponential_example": {
            "class_name": "Exponential",
            "params": {
        """

        template = {}
        for dist_name, dist_class in cls._registry.items():
            # Get the signature of the __init__ method
            signature = inspect.signature(dist_class.__init__)
            params = {}

            # Add parameters (excluding 'self' and 'random_seed')
            for param_name, param in signature.parameters.items():
                if param_name not in ["self", "random_seed"]:
                    # If parameter has a default value and it's not None
                    if param.default is not param.empty and param.default is not None:
                        params[param_name] = param.default
                    else:
                        # Use appropriate dummy values based on parameter name
                        if "mean" in param_name:
                            params[param_name] = 1.0
                        elif any(
                            name in param_name for name in ["std", "scale", "lambda"]
                        ):
                            params[param_name] = 1.0
                        elif any(name in param_name for name in ["low", "min"]):
                            params[param_name] = 0.0
                        elif any(name in param_name for name in ["high", "max"]):
                            params[param_name] = 10.0
                        elif "mode" in param_name:
                            params[param_name] = 5.0
                        elif "shape" in param_name:
                            params[param_name] = 2.0
                        else:
                            # Generic fallback
                            params[param_name] = 1.0

            template[f"{dist_name}_example"] = {
                "class_name": dist_name,
                "params": params,
            }

        if format.lower() == "json":
            return json.dumps(template, indent=indent)
        return template


@DistributionRegistry.register()
class Exponential:
    """
    Exponential distribution implementation.

    A probability distribution that models the time between events in a Poisson
    process, where events occur continuously and independently at a constant
    average rate.

    This class conforms to the Distribution protocol and provides methods to
    sample from an exponential distribution with a specified mean.
    """

    def __init__(
        self,
        mean: float,
        random_seed: Optional[Union[int, SeedSequence]] = None,
    ):
        """
        Initialize an exponential distribution.

        Parameters
        ----------
        mean : float
            The mean of the exponential distribution.
            Must be positive.

        random_seed : Optional[Union[int, SeedSequence]], default=None
            A random seed or SeedSequence to reproduce samples. If None, a
            unique sample sequence is generated.
        """
        validate(mean, "mean", is_numeric, is_positive)
        self.rng = np.random.default_rng(random_seed)
        self.mean = mean

    def __repr__(self):
        return f"Exponential(mean={self.mean})"

    def sample(
        self, size: Optional[Union[int, Tuple[int, ...]]] = None
    ) -> Union[float, NDArray[np.float64]]:
        """
        Generate random samples from the exponential distribution.

        Parameters
        ----------
        size : Optional[Union[int, Tuple[int, ...]]], default=None
            The number/shape of samples to generate:
            - If None: returns a single sample as a float
            - If int: returns a 1-D array with that many samples
            - If tuple of ints: returns an array with that shape

        Returns
        -------
        Union[float, NDArray[np.float64]]
            Random samples from the exponential distribution:
            - A single float when size is None
            - A numpy array of floats with shape determined by size parameter
        """
        return self.rng.exponential(self.mean, size=size)


@DistributionRegistry.register()
class Bernoulli:
    """
    Bernoulli distribution implementation.

    A discrete probability distribution that takes value 1 with probability p
    and value 0 with probability 1-p.

    This class conforms to the Distribution protocol and provides methods to
    sample from a Bernoulli distribution with a specified probability.
    """

    def __init__(
        self, p: float, random_seed: Optional[Union[int, SeedSequence]] = None
    ):
        """
        Initialize a Bernoulli distribution.

        Parameters
        ----------
        p : float
            Probability of drawing a 1. Must be between 0 and 1.

        random_seed : Optional[Union[int, SeedSequence]], default=None
            A random seed or SeedSequence to reproduce samples. If None, a
            unique sample sequence is generated.
        """
        validate(p, "p", is_numeric, is_probability)
        self.rng = np.random.default_rng(random_seed)
        self.p = p

    def __repr__(self):
        return f"Bernoulli(p={self.p})"

    def sample(
        self, size: Optional[Union[int, Tuple[int, ...]]] = None
    ) -> Union[float, NDArray[np.float64]]:
        """
        Generate random samples from the Bernoulli distribution.

        Parameters
        ----------
        size : Optional[Union[int, Tuple[int, ...]]], default=None
            The number/shape of samples to generate:
            - If None: returns a single sample as a float
            - If int: returns a 1-D array with that many samples
            - If tuple of ints: returns an array with that shape

        Returns
        -------
        Union[float, NDArray[np.float64]]
            Random samples from the Bernoulli distribution:
            - A single float (0 or 1) when size is None
            - A numpy array of floats (0s and 1s) with shape determined
              by size parameter
        """
        return self.rng.binomial(n=1, p=self.p, size=size)


@DistributionRegistry.register()
class Lognormal:
    """
    Lognormal distribution implementation.

    A continuous probability distribution where the logarithm of a random
    variable is normally distributed. It is useful for modeling variables that
    are the product of many small independent factors.

    This class conforms to the Distribution protocol and provides methods to
    sample from a lognormal distribution with a specified mean and standard
    deviation.
    """

    def __init__(
        self,
        mean: float,
        stdev: float,
        random_seed: Optional[Union[int, SeedSequence]] = None,
    ):
        """
        Initialize a lognormal distribution.

        Parameters
        ----------
        mean : float
            Mean of the lognormal distribution.

        stdev : float
            Standard deviation of the lognormal distribution.

        random_seed : Optional[Union[int, SeedSequence]], default=None
            A random seed or SeedSequence to reproduce samples. If None, a
            unique sample sequence is generated.
        """
        validate(mean, "mean", is_numeric, is_positive)
        validate(stdev, "stdev", is_numeric, is_positive)
        self.rng = np.random.default_rng(random_seed)
        mu, sigma = self.normal_moments_from_lognormal(mean, stdev**2)
        self.mu = mu
        self.sigma = sigma
        self.mean = mean
        self.stdev = stdev

    def __repr__(self):
        return f"Lognormal(mean={self.mean}, stdev={self.stdev})"

    def normal_moments_from_lognormal(self, m: float, v: float) -> Tuple[float, float]:
        """
        Calculate mu and sigma of the normal distribution underlying
        a lognormal with mean m and variance v.

        Parameters
        ----------
        m : float
            Mean of lognormal distribution.
        v : float
            Variance of lognormal distribution.

        Returns
        -------
        Tuple[float, float]
            The mu and sigma parameters of the underlying normal distribution.

        Notes
        -----
        Formula source:
        https://blogs.sas.com/content/iml/2014/06/04/simulate-lognormal-data-
        with-specified-mean-and-variance.html
        """
        phi = math.sqrt(v + m**2)
        mu = math.log(m**2 / phi)
        sigma = math.sqrt(math.log(phi**2 / m**2))
        return mu, sigma

    def sample(
        self, size: Optional[Union[int, Tuple[int, ...]]] = None
    ) -> Union[float, NDArray[np.float64]]:
        """
        Generate random samples from the lognormal distribution.

        Parameters
        ----------
        size : Optional[Union[int, Tuple[int, ...]]], default=None
            The number/shape of samples to generate:
            - If None: returns a single sample as a float
            - If int: returns a 1-D array with that many samples
            - If tuple of ints: returns an array with that shape

        Returns
        -------
        Union[float, NDArray[np.float64]]
            Random samples from the lognormal distribution:
            - A single float when size is None
            - A numpy array of floats with shape determined by size parameter
        """
        return self.rng.lognormal(self.mu, self.sigma, size=size)


@DistributionRegistry.register()
class Normal:
    """
    Normal distribution implementation with optional truncation.

    A continuous probability distribution that follows the Gaussian bell curve.
    This implementation allows truncating the distribution at a minimum value.

    This class conforms to the Distribution protocol and provides methods to
    sample from a normal distribution with specified mean and standard
    deviation.
    """

    def __init__(
        self,
        mean: float,
        sigma: float,
        minimum: Optional[float] = None,
        random_seed: Optional[Union[int, SeedSequence]] = None,
    ):
        """
        Initialize a normal distribution.

        Parameters
        ----------
        mean : float
            The mean (μ) of the normal distribution.

        sigma : float
            The standard deviation (σ) of the normal distribution.

        minimum : Optional[float], default=None
            If provided, truncates the distribution to this minimum value.
            Any sampled values below this minimum will be set to this value.

        random_seed : Optional[Union[int, SeedSequence]], default=None
            A random seed or SeedSequence to reproduce samples. If None, a
            unique sample sequence is generated.
        """
        validate(mean, "mean", is_numeric)
        validate(sigma, "sigma", is_numeric, is_positive)

        if minimum is not None:
            validate(minimum, "minimum", is_numeric)

        self.rng = np.random.default_rng(seed=random_seed)
        self.mean = mean
        self.sigma = sigma
        self.minimum = minimum

    def __repr__(self):
        if self.minimum is None:
            return f"Normal(mean={self.mean}, sigma={self.sigma})"
        return (
            f"Normal(mean={self.mean}, sigma={self.sigma}, "
            + f"minimum={self.minimum})"
        )

    def sample(
        self, size: Optional[Union[int, Tuple[int, ...]]] = None
    ) -> Union[float, NDArray[np.float64]]:
        """
        Generate random samples from the normal distribution.

        Parameters
        ----------
        size : Optional[Union[int, Tuple[int, ...]]], default=None
            The number/shape of samples to generate:
            - If None: returns a single sample as a float
            - If int: returns a 1-D array with that many samples
            - If tuple of ints: returns an array with that shape

        Returns
        -------
        Union[float, NDArray[np.float64]]
            Random samples from the normal distribution:
            - A single float when size is None
            - A numpy array of floats with shape determined by size parameter

        Notes
        -----
        If a minimum value was specified during initialization, any samples
        below this value will be truncated (set to the minimum value).
        """
        samples = self.rng.normal(self.mean, self.sigma, size=size)

        if self.minimum is None:
            return samples

        if size is None:
            return max(self.minimum, samples)

        # Truncate samples below minimum
        below_min_idx = np.where(samples < self.minimum)[0]
        samples[below_min_idx] = self.minimum
        return samples


@DistributionRegistry.register()
class Uniform:
    """
    Uniform distribution implementation.

    A continuous probability distribution where all values in a range have
    equal probability of being sampled.

    This class conforms to the Distribution protocol and provides methods to
    sample from a uniform distribution between specified low and high values.
    """

    def __init__(
        self,
        low: float,
        high: float,
        random_seed: Optional[Union[int, SeedSequence]] = None,
    ):
        """
        Initialize a uniform distribution.

        Parameters
        ----------
        low : float
            Lower bound of the distribution range.

        high : float
            Upper bound of the distribution range.

        random_seed : Optional[Union[int, SeedSequence]], default=None
            A random seed or SeedSequence to reproduce samples. If None, a
            unique sample sequence is generated.
        """
        validate(low, "low", is_numeric)
        validate(high, "high", is_numeric)
        is_ordered_pair(low, high)
        self.rng = np.random.default_rng(random_seed)
        self.low = low
        self.high = high

    def __repr__(self):
        return f"Uniform(low={self.low}, high={self.high})"

    def sample(
        self, size: Optional[Union[int, Tuple[int, ...]]] = None
    ) -> Union[float, NDArray[np.float64]]:
        """
        Generate random samples from the uniform distribution.

        Parameters
        ----------
        size : Optional[Union[int, Tuple[int, ...]]], default=None
            The number/shape of samples to generate:
            - If None: returns a single sample as a float
            - If int: returns a 1-D array with that many samples
            - If tuple of ints: returns an array with that shape

        Returns
        -------
        Union[float, NDArray[np.float64]]
            Random samples from the uniform distribution:
            - A single float when size is None
            - A numpy array of floats with shape determined by size parameter
        """
        return self.rng.uniform(low=self.low, high=self.high, size=size)


@DistributionRegistry.register()
class Triangular:
    """
    Triangular distribution implementation.

    A continuous probability distribution with lower limit, upper limit, and
    mode, forming a triangular-shaped probability density function.

    This class conforms to the Distribution protocol and provides methods to
    sample from a triangular distribution with specified parameters.
    """

    def __init__(
        self,
        low: float,
        mode: float,
        high: float,
        random_seed: Optional[Union[int, SeedSequence]] = None,
    ):
        """
        Initialize a triangular distribution.

        Parameters
        ----------
        low : float
            Lower limit of the distribution.

        mode : float
            Mode (peak) of the distribution. Must be between low and high.

        high : float
            Upper limit of the distribution.

        random_seed : Optional[Union[int, SeedSequence]], default=None
            A random seed or SeedSequence to reproduce samples. If None, a
            unique sample sequence is generated.
        """

        # validation
        for name, value in [("low", low), ("mode", mode), ("high", high)]:
            validate(value, name, is_numeric)

        is_ordered_triplet(low, mode, high, middle_name="mode")

        self.rng = np.random.default_rng(random_seed)
        self.low = low
        self.high = high
        self.mode = mode

    def __repr__(self):
        return f"Triangular(low={self.low}, mode={self.mode}, high={self.high})"

    def sample(
        self, size: Optional[Union[int, Tuple[int, ...]]] = None
    ) -> Union[float, NDArray[np.float64]]:
        """
        Generate random samples from the triangular distribution.

        Parameters
        ----------
        size : Optional[Union[int, Tuple[int, ...]]], default=None
            The number/shape of samples to generate:
            - If None: returns a single sample as a float
            - If int: returns a 1-D array with that many samples
            - If tuple of ints: returns an array with that shape

        Returns
        -------
        Union[float, NDArray[np.float64]]
            Random samples from the triangular distribution:
            - A single float when size is None
            - A numpy array of floats with shape determined by size parameter
        """
        return self.rng.triangular(self.low, self.mode, self.high, size=size)


@DistributionRegistry.register()
class FixedDistribution:
    """
    Fixed distribution implementation.

    A degenerate distribution that always returns the same fixed value.
    Useful for constants or deterministic parameters in models.

    This class conforms to the Distribution protocol and provides methods to
    sample a constant value regardless of the number of samples requested.
    """

    def __init__(self, value: float):
        """
        Initialize a fixed distribution.

        Parameters
        ----------
        value : float
            The constant value that will be returned by sampling.
        """
        validate(value, "value", is_numeric)
        self.value = value

    def __repr__(self):
        return f"FixedDistribution(value={self.value})"

    def sample(
        self, size: Optional[Union[int, Tuple[int, ...]]] = None
    ) -> Union[float, NDArray[np.float64]]:
        """
        Generate "samples" from the fixed distribution (always the same value).

        Parameters
        ----------
        size : Optional[Union[int, Tuple[int, ...]]], default=None
            The number/shape of samples to generate:
            - If None: returns the fixed value as a float
            - If int: returns a 1-D array filled with the fixed value
            - If tuple of ints: returns an array with that shape filled with
              the fixed value

        Returns
        -------
        Union[float, NDArray[np.float64]]
            The fixed value:
            - A single float when size is None
            - A numpy array filled with the fixed value with shape
              determined by size parameter
        """
        if size is not None:
            return np.full(size, self.value)
        return self.value


@DistributionRegistry.register()
class CombinationDistribution:
    """
    Combination distribution implementation.

    A distribution that combines (sums) samples from multiple underlying
    distributions. Useful for modeling compound effects or building complex
    distributions from simpler ones.

    This class conforms to the Distribution protocol and provides methods to
    sample a combination of values from multiple distributions.
    """

    def __init__(self, *args: Distribution, dists=None):
        """
        Initialise a combination distribution.

        Distributions can be passed either as positional arguments or via the
        `dists` keyword argument, but not both. The keyword form is required
        when creating instances through the `DistributionRegistry` class,
        which passes parameters by name.

        Parameters
        ----------
        *args : Distribution
            Distribution objects to combine, passed as positional arguments.
            E.g. `CombinationDistribution(d1, d2)`.
            Cannot be used together with `dists`.
        dists : Sequence[Distribution], optional
            Distribution objects to combine, passed as a keyword argument.
            E.g. `CombinationDistribution(dists=[d1, d2])`.
            Cannot be used together with `*args`.
        """
        if args and dists is not None:
            raise ValueError(
                "Pass distributions either as positional arguments or as "
                "'dists', not both."
            )
        self.dists = dists if dists is not None else args

    def __repr__(self):
        dist_reprs = [repr(dist) for dist in self.dists]
        return f"CombinationDistribution({', '.join(dist_reprs)})"

    def sample(
        self, size: Optional[Union[int, Tuple[int, ...]]] = None
    ) -> Union[float, NDArray[np.float64]]:
        """
        Generate random samples from the combination distribution.

        For each sample drawn, the result is the sum of samples from each
        of the underlying distributions.

        Parameters
        ----------
        size : Optional[Union[int, Tuple[int, ...]]], default=None
            The number/shape of samples to generate:
            - If None: returns a single combined sample as a float
            - If int: returns a 1-D array with that many combined samples
            - If tuple of ints: returns an array with that shape of combined
              samples

        Returns
        -------
        Union[float, NDArray[np.float64]]
            Random samples from the combination distribution:
            - A single float (sum of component samples) when size is None
            - A numpy array of combined samples with shape determined by size
              parameter
        """
        total = 0.0 if size is None else np.zeros(size)

        for dist in self.dists:
            total += dist.sample(size)
        return total


@DistributionRegistry.register()
class GroupedContinuousEmpirical:
    """
    Continuous Empirical Distribution for Grouped Data implementation.

    A distribution that performs linear interpolation between upper and lower
    bounds of a discrete distribution. Useful for modeling empirical data with
    a continuous approximation.

    This class conforms to the Distribution protocol and provides methods to
    sample from a continuous empirical distribution.
    """

    def __init__(
        self,
        lower_bounds: ArrayLike,
        upper_bounds: ArrayLike,
        freq: ArrayLike,
        random_seed: Optional[Union[int, SeedSequence]] = None,
    ):
        """
        Initialize a continuous empirical distribution.

        Parameters
        ----------
        lower_bounds : ArrayLike
            Lower bounds of a discrete empirical distribution.

        upper_bounds : ArrayLike
            Upper bounds of a discrete empirical distribution.

        freq : ArrayLike
            Frequency of observations between bounds.

        random_seed : Optional[Union[int, SeedSequence]], default=None
            A random seed or SeedSequence to reproduce samples. If None, a
            unique sample sequence is generated.
        """
        self.rng = np.random.default_rng(random_seed)
        self.lower_bounds = np.asarray(lower_bounds)
        self.upper_bounds = np.asarray(upper_bounds)
        self.cumulative_probs = self.create_cumulative_probs(freq)

    def __repr__(self):
        lb_repr = (
            str(self.lower_bounds.tolist())
            if len(self.lower_bounds) < 4
            else f"[{', '.join(str(x) for x in self.lower_bounds[:3])}, ...]"
        )
        ub_repr = (
            str(self.upper_bounds.tolist())
            if len(self.upper_bounds) < 4
            else f"[{', '.join(str(x) for x in self.upper_bounds[:3])}, ...]"
        )
        return (
            f"ContinuousEmpirical(lower_bounds={lb_repr}, "
            + f"upper_bounds={ub_repr}, freq=...)"
        )

    @property
    def mean(self) -> float:
        """Calculate the theoretical mean of the distribution."""
        # Calculate midpoints of each bin
        midpoints = (self.lower_bounds + self.upper_bounds) / 2

        # Get the probabilities from the cumulative probabilities
        probs = np.diff(np.append(0, self.cumulative_probs))

        # Return weighted average of midpoints
        return np.sum(midpoints * probs)

    @property
    def variance(self) -> float:
        """
        Calculate the theoretical variance of the GroupedContinuousEmpirical
        distribution.

        The total variance is composed of two components:
        1. Between-bin variance: Variance arising from the differences between
           bin midpoints
        2. Within-bin variance: Additional variance from the linear
           interpolation within each bin

        For a linear interpolation model, the within-bin component follows the
        variance formula for a uniform distribution: (bin_width)²/12 for each
        bin, weighted by the bin's probability.

        Returns
        -------
        float
            The theoretical variance of the distribution.

        Notes
        -----
        This calculation provides the exact theoretical variance of a
        continuous distribution created through linear interpolation between
        grouped data points. The formula accounts for both the positioning of
        the groups and the additional variance introduced by the
        interpolation process itself.

        Simple variance calculations that only consider bin midpoints
        will underestimate the true variance of the
        interpolated distribution.

        Example
        -------
        >>> dist = GroupedContinuousEmpirical([0, 1, 2], [1, 2, 3],
                                              [10, 20, 30])
        >>> dist.variance()
        0.6388888888888888
        """
        # Calculate midpoints of each bin
        midpoints = (self.lower_bounds + self.upper_bounds) / 2

        # Get the probabilities from the cumulative probabilities
        probs = np.diff(np.append(0, self.cumulative_probs))

        # Calculate mean
        mean_val = np.sum(midpoints * probs)

        # Between-bin variance (using midpoints)
        between_bin_variance = np.sum(probs * (midpoints - mean_val) ** 2)

        # Within-bin variance (from uniform distribution in each bin)
        # For a uniform distribution on [a,b], variance = (b-a)²/12
        bin_widths = self.upper_bounds - self.lower_bounds
        within_bin_variance = np.sum(probs * (bin_widths**2) / 12)

        # Total variance is the sum of both components
        return between_bin_variance + within_bin_variance

    def create_cumulative_probs(self, freq: ArrayLike) -> NDArray[np.float64]:
        """
        Calculate cumulative relative frequency from frequency.

        Parameters
        ----------
        freq : ArrayLike
            Frequency distribution.

        Returns
        -------
        NDArray[np.float64]
            Cumulative relative frequency.
        """
        freq = np.asarray(freq, dtype="float")
        return np.cumsum(freq / freq.sum(), dtype="float")

    def sample(
        self, size: Optional[Union[int, Tuple[int, ...]]] = None
    ) -> Union[float, NDArray[np.float64]]:
        """
        Sample from the Continuous Empirical Distribution.

        Parameters
        ----------
        size : Optional[Union[int, Tuple[int, ...]]], default=None
            The number/shape of samples to generate:
            - If None: returns a single sample as a float
            - If int: returns a 1-D array with that many samples
            - If tuple of ints: returns an array with that shape

        Returns
        -------
        Union[float, NDArray[np.float64]]
            Random samples from the continuous empirical distribution:
            - A single float when size is None
            - A numpy array of floats with shape determined by size parameter
        """
        if size is None:
            size = 1

        # Handle the case where size is a tuple - convert to total number of
        # samples
        total_samples = size if isinstance(size, int) else np.prod(size)

        samples = []
        for _ in range(total_samples):
            # Sample a value u from the uniform(0, 1) distribution
            u = self.rng.random()

            # Obtain lower and upper bounds of a sample from the
            # discrete empirical distribution
            idx = np.searchsorted(self.cumulative_probs, u)
            lb, ub = self.lower_bounds[idx], self.upper_bounds[idx]

            # TM fix 19/04/25 - handle lower bound silent error
            # For idx = 0, use 0.0 as the previous cumulative probability
            prev_cumprob = 0.0 if idx == 0 else self.cumulative_probs[idx - 1]

            # Use linear interpolation of u between
            # the lower and upper bound to obtain a continuous value
            proportion = (u - prev_cumprob) / (
                self.cumulative_probs[idx] - prev_cumprob
            )
            continuous_value = lb + proportion * (ub - lb)

            samples.append(continuous_value)

        if total_samples == 1:
            # .item() ensures returned as python 'float'
            # as opposed to np.float64
            return samples[0].item()

        result = np.asarray(samples)
        # Reshape if size was a tuple
        if isinstance(size, tuple):
            result = result.reshape(size)
        return result


@DistributionRegistry.register()
class RawContinuousEmpirical:
    """
    Continuous Empirical Distribution for Raw Data using Law and Kelton's
    method.

    A distribution that performs linear interpolation between data points
    according to the algorithm described in Law & Kelton's "Simulation Modeling
    and Analysis". The implementation follows a two-step approach:

    1. Generate U ~ Uniform(0, 1), calculate P = (n-1)U, and I = int(P) + 1
    2. Return X_I + (P-I)(X_{I+1} - X_I)

    This approach ensures proper weighting across intervals and is suitable for
    both Monte Carlo and discrete-event simulation applications.

    Maximum and minimum values of the distribution are defined by the data.
    """

    def __init__(
        self,
        data: ArrayLike,
        random_seed: Optional[Union[int, SeedSequence]] = None,
    ):
        """
        Initialize a continuous empirical distribution from raw data.

        Parameters
        ----------
        data : ArrayLike
            Raw data points to create the empirical distribution from.

        random_seed : Optional[Union[int, SeedSequence]], default=None
            A random seed or SeedSequence to reproduce samples. If None, a
            unique sample sequence is generated.
        """
        self.rng = np.random.default_rng(random_seed)

        # Sort the data to create the ECDF
        self.data = np.sort(np.asarray(data, dtype=float))

    def __repr__(self):
        data_repr = (
            str(self.data.tolist())
            if len(self.data) < 4
            else f"[{', '.join(str(x) for x in self.data[:3])}, ...]"
        )
        return f"ContinuousEmpirical(data={data_repr})"

    def sample(
        self, size: Optional[Union[int, Tuple[int, ...]]] = None
    ) -> Union[float, NDArray[np.float64]]:
        """
        Sample from the Continuous Empirical Distribution.

        Parameters
        ----------
        size : Optional[Union[int, Tuple[int, ...]]], default=None
            The number/shape of samples to generate:
            - If None: returns a single sample as a float
            - If int: returns a 1-D array with that many samples
            - If tuple of ints: returns an array with that shape

        Returns
        -------
        Union[float, NDArray[np.float64]]
            Random samples from the continuous empirical distribution
        """
        U = self.rng.random(size)
        n = len(self.data)
        P = (n - 1) * U

        if size is None:
            # single sample

            index = int(P) + 1

            # Handle edge case when I is the last index
            if index >= n - 1:
                # return maximum value
                return self.data[-1]

            frac = P - index
            lower = self.data[index]
            upper = self.data[index + 1]
            return max(lower + frac * (upper - lower), self.data[0])

        index = P.astype(int) + 1
        # array operations
        mask = index >= n - 1
        result = np.empty_like(P, dtype=float)

        # Handle edge cases where index equals n-1
        if np.any(mask):
            result[mask] = self.data[-1]

        # Process normal cases with interpolation
        if np.any(~mask):
            valid_index = index[~mask]
            valid_P = P[~mask]
            frac = valid_P - valid_index
            lower = self.data[valid_index]
            upper = self.data[valid_index + 1]
            result[~mask] = lower + frac * (upper - lower)

        # return clipped to lower value
        return result.clip(min=self.data[0])

    def plotly_ecdf_standard(
        self,
        title: Optional[str] = "Standard Empirical CDF",
        xaxis_title: Optional[str] = "Data Value",
        yaxis_title: Optional[str] = "Cumulative Probability (P(X <= x))",
        line_color: Optional[str] = None,  # e.g., 'blue', '#1f77b4'
        line_width: Optional[float] = None,  # e.g., 2
        trace_name: Optional[str] = "Standard ECDF",
        showlegend: bool = True,
        layout_options: Optional[Dict] = None,
    ) -> go.Figure:
        """
        Plots the standard Empirical Cumulative Distribution Function (ECDF)
        using Plotly, with customization options.

        Parameters
        ----------
        title : Optional[str], default="Standard Empirical CDF"
            The main title of the plot.
        xaxis_title : Optional[str], default="Data Value"
            The title for the x-axis.
        yaxis_title : Optional[str], default="Cumulative Probability (P(X <= x))"
            The title for the y-axis.
        line_color : Optional[str], default=None
            Color of the ECDF line (Plotly default if None). Accepts CSS color
            names, hex codes, etc.
        line_width : Optional[float], default=None
            Width of the ECDF line (Plotly default if None).
        trace_name : Optional[str], default="Standard ECDF"
            Name to display in the legend for this trace.
        showlegend : bool, default=True
            Whether to display the legend.
        layout_options : Optional[Dict], default=None
            A dictionary of additional options to pass to fig.update_layout().

        Returns
        -------
        plotly.graph_objects.Figure
            A Plotly figure object containing the plot.
        """
        n = len(self.data)
        if n == 0:
            fig = go.Figure()
            fig.update_layout(
                title=title,
                xaxis_title=xaxis_title,
                yaxis_title=yaxis_title,
                # Hide axes lines/ticks for empty plot
                xaxis={"visible": False},
                yaxis={"visible": False},
            )
            if layout_options:
                fig.update_layout(layout_options)
            return fig

        # Create the basic ECDF plot
        fig = px.ecdf(
            x=self.data,
            # px.ecdf uses 'y' internally for the probability axis label key
            labels={"x": xaxis_title, "y": yaxis_title},
            title=title,  # Set title via px directly
        )

        # Apply trace customizations
        fig.update_traces(
            # Ensure we target the scatter trace created by px.ecdf
            selector={"type": "scatter"},
            name=trace_name,
            showlegend=showlegend,
            line={
                "color": line_color,  # Plotly handles None: uses default
                "width": line_width,  # Plotly handles None: uses default
            },
        )

        # Apply general layout updates (including potential overrides for
        # titles/labels)
        fig.update_layout(
            xaxis_title=xaxis_title,
            yaxis_title=yaxis_title,
            showlegend=showlegend,
            # Explicitly set again in case user wants to override px default
            # via layout_options
            title=title,
        )
        # Apply any extra custom layout options
        if layout_options:
            fig.update_layout(layout_options)

        return fig

    def plotly_ecdf_linear_interpolation(
        self,
        title: Optional[str] = "Piecewise Linear CDF used by Sampler",
        xaxis_title: Optional[str] = "Data Value",
        yaxis_title: Optional[str] = "Cumulative Probability (Sampler's CDF)",
        line_color: Optional[str] = None,
        line_width: Optional[float] = None,
        marker_symbol: Optional[str] = "circle",
        marker_size: Optional[float] = 6,
        marker_color: Optional[str] = None,
        trace_name: Optional[str] = "Piecewise Linear CDF",
        showlegend: bool = True,
        layout_options: Optional[Dict] = None,
    ) -> go.Figure:
        """
        Plots the piecewise linear CDF implied by the Law & Kelton sampling
        method using Plotly, with customization options.

        Parameters
        ----------
        title : Optional[str], default="Piecewise Linear CDF used by Sampler"
            The main title of the plot.
        xaxis_title : Optional[str], default="Data Value"
            The title for the x-axis.
        yaxis_title : Optional[str], default="Cumulative Probability (Sampler's CDF)"
            The title for the y-axis.
        line_color : Optional[str], default=None
            Color of the line segments (Plotly default if None). Examples:
            "green", "#2ca02c".
        line_width : Optional[float], default=None
            Width of the line segments (Plotly default if None). Example: 2.
        marker_symbol : Optional[str], default='circle'
            Symbol for markers at data points (Plotly default if None). Use
            None to hide markers. Examples: "circle", "square", "x".
        marker_size : Optional[float], default=6
            Size of the markers (Plotly default if None).
        marker_color : Optional[str], default=None
            Color of the markers (inherits from line by default, or specify).
        trace_name : Optional[str], default="Piecewise Linear CDF"
            Name to display in the legend for this trace.
        showlegend : bool, default=True
            Whether to display the legend.
        layout_options : Optional[Dict], default=None
            A dictionary of additional options to pass to fig.update_layout().

        Returns
        -------
        plotly.graph_objects.Figure
            A Plotly figure object containing the plot.
        """
        n = len(self.data)

        # --- Handle Edge Cases (n=0, n=1) ---
        if n == 0:
            fig = go.Figure()
            fig.update_layout(
                title=title,
                xaxis_title=xaxis_title,
                yaxis_title=yaxis_title,
                xaxis={"visible": False},
                yaxis={"visible": False},
            )
            if layout_options:
                fig.update_layout(layout_options)
            return fig

        if n == 1:
            # Plot a vertical line segment using go.Scatter
            fig = go.Figure(
                data=go.Scatter(
                    x=[self.data[0], self.data[0]],
                    y=[0, 1],
                    mode="lines+markers",
                    name=trace_name,
                    line={"color": line_color, "width": line_width},
                    marker={
                        "symbol": marker_symbol,
                        "size": marker_size,
                        "color": marker_color,
                    },
                    showlegend=showlegend,
                )
            )
            fig.update_layout(
                title=title,
                xaxis_title=xaxis_title,
                yaxis_title=yaxis_title,
                showlegend=showlegend,
            )
            # Optional: Adjust x-axis range for visibility if needed
            # fig.update_xaxes(range=[self.data[0] - 1, self.data[0] + 1])
            if layout_options:
                fig.update_layout(layout_options)
            return fig

        # --- Handle General Case (n > 1) ---
        # Calculate points for linear interpolation: (x_i, (i-1)/(n-1))
        x_linear = self.data
        y_linear = np.arange(n) / (n - 1)

        # Create the basic line plot
        # Note: px.line doesn't directly support marker_symbol/size in the call
        # itself - we will apply marker styles via update_traces
        fig = px.line(
            x=x_linear,
            y=y_linear,
            # Enable markers if symbol is specified
            markers=(marker_symbol is not None),
            labels={"x": xaxis_title, "y": yaxis_title},
            title=title,
        )

        # Determine marker color (use line color if marker color not specified)
        final_marker_color = marker_color if marker_color is not None else line_color

        # Apply trace customizations
        fig.update_traces(
            # Target the scatter trace from px.line
            selector={"type": "scatter"},
            name=trace_name,
            showlegend=showlegend,
            line={"color": line_color, "width": line_width},
            marker={
                "symbol": marker_symbol,
                "size": marker_size,
                # Apply potentially derived marker colour
                "color": final_marker_color,
                # You could also add marker line properties here if needed:
                # line=dict(color='black', width=1)
            },
        )

        # Apply general layout updates
        fig.update_layout(
            xaxis_title=xaxis_title,
            yaxis_title=yaxis_title,
            showlegend=showlegend,
            # Ensure title consistency
            title=title,
        )
        if layout_options:
            fig.update_layout(layout_options)

        return fig


@DistributionRegistry.register()
class Erlang:
    """
    Erlang distribution implementation.

    A continuous probability distribution that is a special case of the Gamma
    distribution where the shape parameter is an integer. This implementation
    allows users to specify mean and standard deviation rather than shape (k)
    and scale (theta) parameters.

    This class conforms to the Distribution protocol and provides methods to
    sample from an Erlang distribution with specified parameters.

    Notes
    -----
    The Erlang is a special case of the gamma distribution where k is an
    integer. Internally this is implemented using numpy Generator's gamma
    method. The k parameter is calculated from the mean and standard deviation
    and rounded to an integer.

    Sources
    -------
    Conversion between mean+stdev to k+theta:
    https://www.statisticshowto.com/erlang-distribution/
    """

    def __init__(
        self,
        mean: float,
        stdev: float,
        location: float = 0.0,
        random_seed: Optional[Union[int, SeedSequence]] = None,
    ):
        """
        Initialize an Erlang distribution.

        Parameters
        ----------
        mean : float
            Mean of the Erlang distribution.

        stdev : float
            Standard deviation of the Erlang distribution.

        location : float, default=0.0
            Offset the origin of the distribution. The returned value
            will be the sampled value plus this location parameter.

        random_seed : Optional[Union[int, SeedSequence]], default=None
            A random seed or SeedSequence to reproduce samples. If None, a
            unique sample sequence is generated.
        """
        for name, value in [("mean", mean), ("stdev", stdev)]:
            validate(value, name, is_numeric, is_positive)

        validate(location, "location", is_numeric, is_non_negative)

        self.rng = np.random.default_rng(random_seed)
        self.mean = mean
        self.stdev = stdev
        self.location = location

        # k also referred to as shape
        self.k = round((mean / stdev) ** 2)

        # theta also referred to as scale
        self.theta = mean / self.k

    def __repr__(self):
        if self.location == 0.0:
            return f"Erlang(mean={self.mean}, stdev={self.stdev})"
        return (
            f"Erlang(mean={self.mean}, stdev={self.stdev}, "
            + f"location={self.location})"
        )

    def sample(
        self, size: Optional[Union[int, Tuple[int, ...]]] = None
    ) -> Union[float, NDArray[np.float64]]:
        """
        Generate random samples from the Erlang distribution.

        Parameters
        ----------
        size : Optional[Union[int, Tuple[int, ...]]], default=None
            The number/shape of samples to generate:
            - If None: returns a single sample as a float
            - If int: returns a 1-D array with that many samples
            - If tuple of ints: returns an array with that shape

        Returns
        -------
        Union[float, NDArray[np.float64]]
            Random samples from the Erlang distribution:
            - A single float when size is None
            - A numpy array of floats with shape determined by size parameter
        """
        return self.rng.gamma(self.k, self.theta, size) + self.location


@DistributionRegistry.register()
class Weibull:
    """
    Weibull distribution implementation.

    A continuous probability distribution useful for modeling time-to-failure
    and similar phenomena. Characterized by shape (alpha) and scale (beta)
    parameters.

    This implementation also includes a third parameter "location" (default=0)
    to shift the distribution if a lower bound is needed.

    The probability density function (PDF) is:
    f(x) = (α/β) * ((x-location)/β)^(α-1) * exp(-((x-location)/β)^α)
    for x ≥ location, where α is the shape parameter and β is the scale
    parameter.

    The samples are generated using:
    X = scale × (-ln(U))^(1/shape) + location
    where U is a uniform random number between 0 and 1.

    The Weibull distribution reduces to:
    - Exponential distribution when shape=1
    - Rayleigh distribution when shape=2
    - Approximately Normal distribution when shape≈3.4
    """

    def __init__(
        self,
        alpha: float,
        beta: float,
        location: float = 0.0,
        random_seed: Optional[Union[int, SeedSequence]] = None,
    ):
        """
        Initialize a three-parameter Weibull distribution.

        Parameters
        ----------
        alpha : float
            The shape parameter. Must be > 0.

        beta : float
            The scale parameter. Must be > 0. The higher the scale parameter,
            the more variance in the samples.

        location : float, default=0.0
            An offset to shift the distribution from 0.

        random_seed : Optional[Union[int, SeedSequence]], default=None
            A random seed or SeedSequence to reproduce samples. If None, a
            unique sample sequence is generated.

        Notes
        -----
        Caution is advised when setting shape and scale parameters as different
        sources use different notations:

        - In Law and Kelton, shape=alpha and scale=beta
        - Wikipedia defines shape=k and scale=lambda=1/beta
        - Other sources define shape=beta and scale=eta (η)
        - In Python's random.weibullvariate, alpha=scale and beta=shape!

        It's recommended to verify the mean and variance of samples match
        expectations.
        """

        validate(alpha, "alpha", is_numeric, is_positive)
        validate(beta, "beta", is_numeric, is_positive)
        validate(location, "location", is_numeric, is_non_negative)

        self.rng = np.random.default_rng(random_seed)
        self.shape = alpha
        self.scale = beta
        self.location = location

    def __repr__(self):
        if self.location == 0.0:
            return f"Weibull(alpha={self.shape}, beta={self.scale})"
        return (
            f"Weibull(alpha={self.shape}, beta={self.scale}, "
            + f"location={self.location})"
        )

    @property
    def mean(self) -> float:
        """
        Return the theoretical mean of the Weibull distribution.

        The formula is: location + scale * Γ(1 + 1/shape)
        where Γ is the gamma function.

        Returns
        -------
        float
            The theoretical mean value of the distribution.
        """
        return self.location + self.scale * math.gamma(1 + 1 / self.shape)

    @property
    def variance(self) -> float:
        """
        Return the theoretical variance of the Weibull distribution.

        The formula is: scale² * [Γ(1 + 2/shape) - (Γ(1 + 1/shape))²]
        where Γ is the gamma function.

        Returns
        -------
        float
            The theoretical variance of the distribution.
        """
        mean_term = math.gamma(1 + 1 / self.shape)
        variance_term = math.gamma(1 + 2 / self.shape)
        return (self.scale**2) * (variance_term - mean_term**2)

    def sample(
        self, size: Optional[Union[int, Tuple[int, ...]]] = None
    ) -> Union[float, NDArray[np.float64]]:
        """
        Generate random samples from the Weibull distribution.

        Parameters
        ----------
        size : Optional[Union[int, Tuple[int, ...]]], default=None
            The number/shape of samples to generate:
            - If None: returns a single sample as a float
            - If int: returns a 1-D array with that many samples
            - If tuple of ints: returns an array with that shape

        Returns
        -------
        Union[float, NDArray[np.float64]]
            Random samples from the Weibull distribution:
            - A single float when size is None
            - A numpy array of floats with shape determined by size parameter
        """
        return self.scale * self.rng.weibull(self.shape, size) + self.location


@DistributionRegistry.register()
class Gamma:
    """
    Gamma distribution implementation with shape (alpha) and scale (beta)
    parameters.

    This class conforms to the Distribution protocol and provides methods to:
    - Calculate theoretical mean and variance
    - Derive parameters from specified mean/variance
    - Generate samples from the distribution
    """

    def __init__(
        self,
        alpha: float,
        beta: float,
        location: float = 0.0,
        random_seed: Optional[Union[int, SeedSequence]] = None,
    ):
        """
        Initialize a Gamma distribution.

        Parameters
        ----------
        alpha : float
            Shape parameter. Must be > 0.

        beta : float
            Scale parameter. Must be > 0.

        location : float, default=0.0
            Offset value added to all samples.

        random_seed : Optional[Union[int, SeedSequence]], default=None
            A random seed or SeedSequence to reproduce samples. If None, a
            unique sample sequence is generated.

        Raises
        ------
        ValueError
            If alpha or beta or location are not positive.
        """
        validate(alpha, "alpha", is_numeric, is_positive)
        validate(beta, "beta", is_numeric, is_positive)
        validate(location, "location", is_numeric, is_non_negative)

        self.rng = np.random.default_rng(random_seed)
        self.alpha = alpha  # shape
        self.beta = beta  # scale
        self.location = location

    def __repr__(self):
        if self.location == 0.0:
            return f"Gamma(alpha={self.alpha}, beta={self.beta})"
        return (
            f"Gamma(alpha={self.alpha}, beta={self.beta}, "
            + f"location={self.location})"
        )

    @property
    def mean(self) -> float:
        """
        Calculate the theoretical mean of the distribution.

        Returns
        -------
        float
            Mean value: α * β
        """
        return self.alpha * self.beta

    @property
    def variance(self) -> float:
        """
        Calculate the theoretical variance of the distribution.

        Returns
        -------
        float
            Variance value: α * β²
        """
        return self.alpha * (self.beta**2)

    @staticmethod
    def params_from_mean_and_var(mean: float, var: float) -> Tuple[float, float]:
        """
        Derive shape (α) and scale (β) parameters from mean and variance.

        Parameters
        ----------
        mean : float
            Target mean value (μ)
        var : float
            Target variance value (σ²)

        Returns
        -------
        Tuple[float, float]
            (alpha, beta) parameters

        Notes
        -----
        Uses formulae:
        - α = μ²/σ²
        - β = σ²/μ
        """
        alpha = mean**2 / var
        beta = var / mean
        return alpha, beta

    def sample(
        self, size: Optional[Union[int, Tuple[int, ...]]] = None
    ) -> Union[float, NDArray[np.float64]]:
        """
        Generate random samples from the Gamma distribution.

        Parameters
        ----------
        size : Optional[Union[int, Tuple[int, ...]]], default=None
            Output shape:
            - None returns single sample as float
            - int returns 1D array
            - tuple returns nD array

        Returns
        -------
        Union[float, NDArray[np.float64]]
            Samples with specified shape, plus location offset
        """
        samples = self.rng.gamma(self.alpha, self.beta, size)
        return samples + self.location


@DistributionRegistry.register()
class Beta:
    """
    Beta distribution implementation.

    A flexible continuous probability distribution defined on the interval
    [0,1], which can be rescaled to any arbitrary interval [min, max].

    As defined in Simulation Modeling and Analysis (Law, 2007).

    Common uses:
    -----------
    1. Useful as a rough model in the absence of data
    2. Distribution of a random proportion
    3. Time to complete a task
    """

    def __init__(
        self,
        alpha1: float,
        alpha2: float,
        lower_bound: float = 0.0,
        upper_bound: float = 1.0,
        random_seed: Optional[Union[int, SeedSequence]] = None,
    ):
        """
        Initialize a Beta distribution.

        Parameters
        ----------
        alpha1 : float
            First shape parameter. Must be positive.

        alpha2 : float
            Second shape parameter. Must be positive.

        lower_bound : float, default=0.0
            Lower bound for rescaling the distribution from [0,1] to
            [lower_bound, upper_bound].

        upper_bound : float, default=1.0
            Upper bound for rescaling the distribution from [0,1] to
            [lower_bound, upper_bound].

        random_seed : Optional[Union[int, SeedSequence]], default=None
            A random seed or SeedSequence to reproduce samples. If None, a
            unique sample sequence is generated.
        """

        # 1. Validate shape parameters
        validate(alpha1, "alpha1", is_numeric, is_positive)
        validate(alpha2, "alpha2", is_numeric, is_positive)

        # 2. Validate bounds
        validate(lower_bound, "lower_bound", is_numeric)
        validate(upper_bound, "upper_bound", is_numeric)

        # 3. Validate relationship between bounds
        is_ordered_pair(lower_bound, upper_bound, "lower_bound", "upper_bound")

        self.rng = np.random.default_rng(random_seed)
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.min = lower_bound
        self.max = upper_bound

    def __repr__(self):
        if self.min == 0.0 and self.max == 1.0:
            return f"Beta(alpha1={self.alpha1}, alpha2={self.alpha2})"
        return (
            f"Beta(alpha1={self.alpha1}, alpha2={self.alpha2}, "
            + f"lower_bound={self.min}, upper_bound={self.max})"
        )

    def sample(
        self, size: Optional[Union[int, Tuple[int, ...]]] = None
    ) -> Union[float, NDArray[np.float64]]:
        """
        Generate random samples from the Beta distribution.

        Parameters
        ----------
        size : Optional[Union[int, Tuple[int, ...]]], default=None
            The number/shape of samples to generate:
            - If None: returns a single sample as a float
            - If int: returns a 1-D array with that many samples
            - If tuple of ints: returns an array with that shape

        Returns
        -------
        Union[float, NDArray[np.float64]]
            Random samples from the Beta distribution, rescaled to [min, max]:
            - A single float when size is None
            - A numpy array of floats with shape determined by size parameter
        """
        return self.min + (
            (self.max - self.min) * self.rng.beta(self.alpha1, self.alpha2, size)
        )


@DistributionRegistry.register()
class DiscreteEmpirical:
    """
    DiscreteEmpirical distribution implementation.

    A probability distribution that samples values with specified frequencies.
    Useful for modeling categorical data or discrete outcomes with known
    probabilities.

    Example uses:
    -------------
    1. Routing percentages
    2. Classes of entity
    3. Batch sizes of arrivals
    """

    def __init__(
        self,
        values: ArrayLike,
        freq: ArrayLike,
        random_seed: Optional[Union[int, SeedSequence]] = None,
    ):
        """
        Initialize a discrete distribution.

        Parameters
        ----------
        values : ArrayLike
            List of possible outcome values. Must be of equal length to freq.

        freq : ArrayLike
            List of observed frequencies or probabilities. Must be of equal
            length to values. These will be normalized to sum to 1.

        random_seed : Optional[Union[int, SeedSequence]], default=None
            A random seed or SeedSequence to reproduce samples. If None, a
            unique sample sequence is generated.

        Raises
        ------
        TypeError
            If freq is not a positive array.
        ValueError
            If values and freq have different lengths.
        """

        # convert to array first
        self.values = np.asarray(list(values))
        self.freq = np.asarray(freq)

        validate(self.freq, "freq", is_positive_array)

        if len(self.values) != len(self.freq):
            raise ValueError("values and freq arguments must be of equal length")

        self.rng = np.random.default_rng(random_seed)
        self.probabilities = self.freq / self.freq.sum()

    def __repr__(self):
        values_repr = (
            str(self.values.tolist())
            if len(self.values) < 4
            else f"[{', '.join(str(x) for x in self.values[:3])}, ...]"
        )
        freq_repr = (
            str(self.freq.tolist())
            if len(self.freq) < 4
            else f"[{', '.join(str(x) for x in self.freq[:3])}, ...]"
        )
        return f"Discrete(values={values_repr}, freq={freq_repr})"

    def sample(
        self, size: Optional[Union[int, Tuple[int, ...]]] = None
    ) -> Union[Any, NDArray]:
        """
        Generate random samples from the discrete distribution.

        Parameters
        ----------
        size : Optional[Union[int, Tuple[int, ...]]], default=None
            The number/shape of samples to generate:
            - If None: returns a single sample
            - If int: returns a 1-D array with that many samples
            - If tuple of ints: returns an array with that shape

        Returns
        -------
        Union[Any, NDArray]
            Random samples from the discrete distribution:
            - A single value (of whatever type was in the values array) when
              size is None
            - A numpy array of values with shape determined by size parameter
        """
        sample = self.rng.choice(self.values, p=self.probabilities, size=size)
        if size is None:
            return sample.item()
        return sample


@DistributionRegistry.register()
class TruncatedDistribution:
    """
    Truncated Distribution implementation.

    Wraps any distribution conforming to the Distribution protocol and
    truncates samples at a specified lower bound. No resampling is performed;
    the class simply ensures no values are below the lower bound.

    This class itself conforms to the Distribution protocol.
    """

    def __init__(self, dist_to_truncate: Distribution, lower_bound: float):
        """
        Initialize a truncated distribution.

        Parameters
        ----------
        dist_to_truncate : Distribution
            Any object conforming to the Distribution protocol that generates
            samples.

        lower_bound : float
            Truncation point. Any samples below this value will be set to this
            value.
        """
        validate(lower_bound, is_numeric)
        self.dist = dist_to_truncate
        self.lower_bound = lower_bound

    def __repr__(self):
        return (
            f"TruncatedDistribution(dist_to_truncate={repr(self.dist)}, "
            + f"lower_bound={self.lower_bound})"
        )

    def sample(
        self, size: Optional[Union[int, Tuple[int, ...]]] = None
    ) -> Union[float, NDArray[np.float64]]:
        """
        Generate random samples from the truncated distribution.

        Parameters
        ----------
        size : Optional[Union[int, Tuple[int, ...]]], default=None
            The number/shape of samples to generate:
            - If None: returns a single sample as a float
            - If int: returns a 1-D array with that many samples
            - If tuple of ints: returns an array with that shape

        Returns
        -------
        Union[float, NDArray[np.float64]]
            Random samples from the truncated distribution:
            - A single float when size is None
            - A numpy array of floats with shape determined by size parameter

        Notes
        -----
        All values will be greater than or equal to the specified lower bound.
        """
        if size is None:
            sample = self.dist.sample()
            return max(self.lower_bound, sample)

        samples = self.dist.sample(size)
        if isinstance(samples, np.ndarray):
            samples[samples < self.lower_bound] = self.lower_bound

        return samples


@DistributionRegistry.register()
class RawDiscreteEmpirical:
    """
    Raw Empirical distribution implementation.

    Samples with replacement from a list of empirical values. Useful when no
    theoretical distribution fits the observed data well.

    This class conforms to the Distribution protocol.
    """

    def __init__(
        self,
        values: ArrayLike,
        random_seed: Optional[Union[int, SeedSequence]] = None,
    ):
        """
        Initialize a raw empirical distribution.

        Parameters
        ----------
        values : ArrayLike
            List of empirical sample values to sample from with replacement.

        random_seed : Optional[Union[int, SeedSequence]], default=None
            A random seed or SeedSequence to reproduce samples. If None, a
            unique sample sequence is generated.

        Notes
        -----
        If the sample size is small, consider whether the upper and lower
        limits in the raw data are representative of the real-world system.
        """
        self.rng = np.random.default_rng(random_seed)
        self.values = np.asarray(values)

    def __repr__(self):
        values_repr = (
            str(self.values.tolist())
            if len(self.values) < 4
            else f"[{', '.join(str(x) for x in self.values[:3])}, ...]"
        )
        return f"RawEmpirical(values={values_repr})"

    @property
    def mean(self) -> float:
        """Calculate the theoretical mean of the distribution."""
        return np.mean(self.values)

    @property
    def variance(self) -> float:
        """Calculate the theoretical variance of the distribution."""
        return np.var(self.values, ddof=0)  # ddof=0 for population variance

    def sample(
        self, size: Optional[Union[int, Tuple[int, ...]]] = None
    ) -> Union[Any, NDArray]:
        """
        Generate random samples from the raw empirical data with replacement.

        Parameters
        ----------
        size : Optional[Union[int, Tuple[int, ...]]], default=None
            The number/shape of samples to generate:
            - If None: returns a single sample
            - If int: returns a 1-D array with that many samples
            - If tuple of ints: returns an array with that shape

        Returns
        -------
        Union[Any, NDArray]
            Random samples from the empirical data:
            - A single value when size is None
            - A numpy array of values with shape determined by size parameter
        """
        samples = self.rng.choice(self.values, size)

        # Ensure we return a scalar if size is None, not an array with one
        # element
        if size is None:
            return samples.item()
        return samples


@DistributionRegistry.register()
class PearsonV:
    """
    Pearson Type V distribution implementation (inverse Gamma distribution).

    Where alpha = shape, and beta = scale (both > 0).

    Law (2007, pg 293-294) defines the distribution as
    PearsonV(alpha, beta) = 1/Gamma(alpha, 1/beta) and notes that the
    PDF is similar to that of lognormal, but has a larger spike
    close to 0. It can be used to model the time to complete a task.

    For certain values of the shape parameter the mean and variance can be
    directly computed:

    mean = beta / (alpha - 1) for alpha > 1.0
    var = beta^2 / (alpha - 1)^2 × (alpha - 2) for alpha > 2.0

    This class conforms to the Distribution protocol.

    Alternative Sources:
    --------------------
    [1] https://riskwiki.vosesoftware.com/PearsonType5distribution.php
    [2] https://modelassist.epixanalytics.com/display/EA/Pearson+Type+5

    Note
    ----
    A good R package for Pearson distributions is PearsonDS
    https://www.rdocumentation.org/packages/PearsonDS/versions/1.3.0
    """

    def __init__(
        self,
        alpha: float,
        beta: float,
        random_seed: Optional[Union[int, SeedSequence]] = None,
    ):
        """
        Initialize a Pearson Type V distribution.

        Parameters
        ----------
        alpha : float
            Shape parameter. Must be > 0.

        beta : float
            Scale parameter. Must be > 0.

        random_seed : Optional[Union[int, SeedSequence]], default=None
            A random seed or SeedSequence to reproduce samples. If None, a
            unique sample sequence is generated.

        Raises
        ------
        ValueError
            If alpha or beta are not positive.
        """
        validate(alpha, "alpha", is_numeric, is_positive)
        validate(beta, "beta", is_numeric, is_positive)

        self.rng = np.random.default_rng(random_seed)
        self.alpha = alpha  # shape
        self.beta = beta  # scale

    def __repr__(self):
        return f"PearsonV(alpha={self.alpha}, beta={self.beta})"

    @property
    def mean(self) -> float:
        """
        Calculate the mean of the Pearson Type V distribution.

        Returns
        -------
        float
            The theoretical mean of this distribution.

        Raises
        ------
        ValueError
            If alpha <= 1.0, as the mean is not defined in this case.
        """
        if self.alpha > 1.0:
            return self.beta / (self.alpha - 1)
        msg = "Cannot directly compute mean when alpha <= 1.0"
        raise ValueError(msg)

    @property
    def variance(self) -> float:
        """
        Calculate the variance of the Pearson Type V distribution.

        Returns
        -------
        float
            The theoretical variance of this distribution.

        Raises
        ------
        ValueError
            If alpha <= 2.0, as the variance is not defined in this case.
        """
        if self.alpha > 2.0:
            return (self.beta**2) / (((self.alpha - 1) ** 2) * (self.alpha - 2))
        msg = "Cannot directly compute var when alpha <= 2.0"
        raise ValueError(msg)

    def sample(
        self, size: Optional[Union[int, Tuple[int, ...]]] = None
    ) -> Union[float, NDArray[np.float64]]:
        """
        Generate random samples from the Pearson Type V distribution.

        Parameters
        ----------
        size : Optional[Union[int, Tuple[int, ...]]], default=None
            The number/shape of samples to generate:
            - If None: returns a single sample as a float
            - If int: returns a 1-D array with that many samples
            - If tuple of ints: returns an array with that shape

        Returns
        -------
        Union[float, NDArray[np.float64]]
            Random samples from the Pearson Type V distribution:
            - A single float when size is None
            - A numpy array of floats with shape determined by size parameter
        """
        return 1 / self.rng.gamma(self.alpha, 1 / self.beta, size)


@DistributionRegistry.register()
class PearsonVI:
    """
    Pearson Type VI distribution implementation (inverted beta distribution).

    Where:
    - alpha1 = shape parameter 1 (> 0)
    - alpha2 = shape parameter 2 (> 0)
    - beta = scale (> 0)

    Law (2007, pg 294-295) notes that PearsonVI can be used to model
    the time to complete a task.

    For certain values of the shape parameters, the mean and variance can be
    directly computed. See functions mean() and var() for details.

    Sampling:
    ---------
    Pearson6(a1,a2,b) = b*X/(1-X), where X=Beta(a1,a2)

    This class conforms to the Distribution protocol.

    Sources:
    --------
    [1] https://riskwiki.vosesoftware.com/PearsonType6distribution.php

    Note
    ----
    A good R package for Pearson distributions is PearsonDS
    https://www.rdocumentation.org/packages/PearsonDS/versions/1.3.0
    """

    def __init__(
        self,
        alpha1: float,
        alpha2: float,
        beta: float,
        random_seed: Optional[Union[int, SeedSequence]] = None,
    ):
        """
        Initialize a Pearson Type VI distribution.

        Parameters
        ----------
        alpha1 : float
            Shape parameter 1. Must be > 0.

        alpha2 : float
            Shape parameter 2. Must be > 0.

        beta : float
            Scale parameter. Must be > 0.

        random_seed : Optional[Union[int, SeedSequence]], default=None
            A random seed or SeedSequence to reproduce samples. If None, a
            unique sample sequence is generated.

        Raises
        ------
        ValueError
            If any of the parameters are not positive.
        """
        validate(alpha1, "alpha1", is_numeric, is_positive)
        validate(alpha1, "alpha2", is_numeric, is_positive)
        validate(beta, "beta", is_numeric, is_positive)

        self.rng = np.random.default_rng(random_seed)
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.beta = beta

    def __repr__(self):
        return (
            f"PearsonVI(alpha1={self.alpha1}, alpha2={self.alpha2}, "
            + f"beta={self.beta})"
        )

    @property
    def mean(self) -> float:
        """
        Calculate the mean of the Pearson Type VI distribution.

        Returns
        -------
        float
            The theoretical mean of this distribution.

        Raises
        ------
        ValueError
            If alpha2 <= 1.0, as the mean is not defined in this case.
        """
        if self.alpha2 > 1.0:
            return (self.beta * self.alpha1) / (self.alpha2 - 1)
        raise ValueError("Cannot compute mean when alpha2 <= 1.0")

    @property
    def variance(self) -> float:
        """
        Calculate the variance of the Pearson Type VI distribution.

        Returns
        -------
        float
            The theoretical variance of this distribution.

        Raises
        ------
        ValueError
            If alpha2 <= 2.0, as the variance is not defined in this case.
        """
        if self.alpha2 > 2.0:
            return ((self.beta**2) * self.alpha1 * (self.alpha1 + self.alpha2 - 1)) / (
                ((self.alpha2 - 1) ** 2) * (self.alpha2 - 2)
            )
        msg = "Cannot directly compute var when alpha2 <= 2.0"
        raise ValueError(msg)

    def sample(
        self, size: Optional[Union[int, Tuple[int, ...]]] = None
    ) -> Union[float, NDArray[np.float64]]:
        """
        Generate random samples from the Pearson Type VI distribution.

        Parameters
        ----------
        size : Optional[Union[int, Tuple[int, ...]]], default=None
            The number/shape of samples to generate:
            - If None: returns a single sample as a float
            - If int: returns a 1-D array with that many samples
            - If tuple of ints: returns an array with that shape

        Returns
        -------
        Union[float, NDArray[np.float64]]
            Random samples from the Pearson Type VI distribution:
            - A single float when size is None
            - A numpy array of floats with shape determined by size parameter
        """
        # Pearson6(a1,a2,b)=b∗X/(1−X), where X=Beta(a1,a2,1)
        x = self.rng.beta(self.alpha1, self.alpha2, size)
        return self.beta * x / (1 - x)


@DistributionRegistry.register()
class ErlangK:
    """
    Erlang distribution where k and theta are specified.

    The Erlang is a special case of the gamma distribution where
    k is a positive integer. Internally this is implemented using
    numpy Generator's gamma method.

    Optionally a user can offset the origin of the distribution
    using the location parameter.
    """

    def __init__(
        self,
        k: int,
        theta: float,
        location: float = 0.0,
        random_seed: Optional[Union[int, SeedSequence]] = None,
    ):
        """
        Initialize an Erlang distribution with specified k and theta.

        Parameters
        ----------
        k : int
            Shape parameter (positive integer) of the Erlang distribution.

        theta : float
            Scale parameter of the Erlang distribution.

        location : float, default=0.0
            Offset the origin of the distribution i.e.
            the returned value = sample[Erlang] + location

        random_seed : Optional[Union[int, SeedSequence]], default=None
            A random seed or SeedSequence to reproduce samples. If None, a
            unique sample sequence is generated.

        Raises
        ------
        ValueError
            If k is not a positive integer.
        """
        # Check that k is a positive integer
        if not isinstance(k, int):
            raise ValueError("k must be an integer")
        if k <= 0:
            raise ValueError("k must be > 0")

        validate(k, "k", is_numeric, is_integer)
        validate(theta, "theta", is_numeric, is_positive)
        validate(location, "location", is_numeric, is_non_negative)

        self.rng = np.random.default_rng(random_seed)
        self.k = k
        self.theta = theta
        self.location = location

    def __repr__(self):
        if self.location == 0.0:
            return f"ErlangK(k={self.k}, theta={self.theta})"
        return (
            f"ErlangK(k={self.k}, theta={self.theta}, " + f"location={self.location})"
        )

    @property
    def mean(self) -> float:
        """Theoretical mean of the Erlang-K distribution."""
        return self.k * self.theta + self.location

    @property
    def variance(self) -> float:
        """Theoretical variance of the Erlang-K distribution."""
        return self.k * (self.theta**2)

    def sample(
        self, size: Optional[Union[int, Tuple[int, ...]]] = None
    ) -> Union[float, NDArray[np.float64]]:
        """
        Generate random samples from the Erlang distribution.

        Parameters
        ----------
        size : Optional[Union[int, Tuple[int, ...]]], default=None
            The number/shape of samples to generate:
            - If None: returns a single sample as a float
            - If int: returns a 1-D array with that many samples
            - If tuple of ints: returns an array with that shape

        Returns
        -------
        Union[float, NDArray[np.float64]]
            Random samples from the Erlang distribution:
            - A single float when size is None
            - A numpy array of floats with shape determined by size parameter
        """
        return self.rng.gamma(self.k, self.theta, size) + self.location


@DistributionRegistry.register()
class Poisson:
    """
    Poisson distribution implementation.

    Used to simulate number of events that occur in an interval of time.
    E.g. number of items in a batch.

    This class conforms to the Distribution protocol.

    Sources:
    --------
    Law (2007 pg. 308) Simulation modelling and analysis.
    """

    def __init__(
        self,
        rate: float,
        random_seed: Optional[Union[int, SeedSequence]] = None,
    ):
        """
        Initialize a Poisson distribution.

        Parameters
        ----------
        rate : float
            Mean number of events in time period.

        random_seed : Optional[Union[int, SeedSequence]], default=None
            A random seed or SeedSequence to reproduce samples. If None, a
            unique sample sequence is generated.
        """
        validate(rate, "rate", is_numeric, is_positive)
        self.rng = np.random.default_rng(random_seed)
        self.rate = rate

    def __repr__(self):
        return f"Poisson(rate={self.rate})"

    def sample(
        self, size: Optional[Union[int, Tuple[int, ...]]] = None
    ) -> Union[int, NDArray[np.int_]]:
        """
        Generate random samples from the Poisson distribution.

        Parameters
        ----------
        size : Optional[Union[int, Tuple[int, ...]]], default=None
            The number/shape of samples to generate:
            - If None: returns a single sample as an integer
            - If int: returns a 1-D array with that many samples
            - If tuple of ints: returns an array with that shape

        Returns
        -------
        Union[int, NDArray[np.int_]]
            Random samples from the Poisson distribution:
            - A single integer when size is None
            - A numpy array of integers with shape determined by size parameter
        """
        return self.rng.poisson(self.rate, size)


@DistributionRegistry.register()
class Hyperexponential:
    """
    Hyperexponential distribution implementation.

    A continuous probability distribution that is a mixture (weighted sum) of
    exponential distributions. It has a higher coefficient of variation than
    a single exponential distribution, making it useful for modeling highly
    variable processes or heavy-tailed phenomena.

    The hyperexponential distribution is useful to model service processes
    where customers may require fundamentally different types of service
    with varying durations. For example, in a technical support call center,
    customers might either:

    1. Have a simple issue (resolved quickly with rate λ₁) with probability p₁

    2. Have a complex issue (requiring longer service with rate λ₂) with
    probability p₂

    This class conforms to the Distribution protocol and provides methods to
    sample from a hyperexponential distribution with specified phase
    probabilities and rates.
    """

    def __init__(
        self,
        probs: ArrayLike,
        rates: ArrayLike,
        random_seed: Optional[Union[int, SeedSequence]] = None,
    ):
        """
        Initialize a hyperexponential distribution.

        Parameters
        ----------
        probs : ArrayLike
            The probabilities (weights) of selecting each exponential
            component. Must sum to 1.0.

        rates : ArrayLike
            The rate parameters for each exponential component.
            Must be positive and same length as probs.

        random_seed : Optional[Union[int, SeedSequence]], default=None
            A random seed or SeedSequence to reproduce samples. If None, a
            unique sample sequence is generated.
        """
        self.rng = np.random.default_rng(random_seed)

        # Convert to numpy arrays
        self.probs = np.asarray(probs, dtype=float)
        self.rates = np.asarray(rates, dtype=float)

        # Validate inputs
        validate(self.probs, "probs", is_probability_vector)
        validate(self.rates, "rates", is_positive_array)

        if len(self.probs) != len(self.rates):
            raise ValueError("probs and rates must have the same length")

    def __repr__(self):
        """
        Return a string representation of the distribution.
        """
        probs_repr = (
            str(self.probs.tolist())
            if len(self.probs) < 4
            else f"[{', '.join(str(p) for p in self.probs[:3])}, ...]"
        )
        rates_repr = (
            str(self.rates.tolist())
            if len(self.rates) < 4
            else f"[{', '.join(str(r) for r in self.rates[:3])}, ...]"
        )
        return f"Hyperexponential(probs={probs_repr}, rates={rates_repr})"

    @property
    def mean(self) -> float:
        """
        Calculate the theoretical mean of the distribution.

        Returns
        -------
        float
            Mean value: sum(p_i / λ_i)
        """
        return np.sum(self.probs / self.rates)

    @property
    def variance(self) -> float:
        """
        Calculate the theoretical variance of the distribution.

        Returns
        -------
        float
            Variance value: 2 * sum(p_i / λ_i^2) - [sum(p_i / λ_i)]^2
        """
        mean = self.mean()
        second_moment = 2 * np.sum(self.probs / (self.rates**2))
        return second_moment - mean**2

    def sample(
        self, size: Optional[Union[int, Tuple[int, ...]]] = None
    ) -> Union[float, NDArray[np.float64]]:
        """
        Generate random samples from the hyperexponential distribution.

        Parameters
        ----------
        size : Optional[Union[int, Tuple[int, ...]]], default=None
            The number/shape of samples to generate:
            - If None: returns a single sample as a float
            - If int: returns a 1-D array with that many samples
            - If tuple of ints: returns an array with that shape

        Returns
        -------
        Union[float, NDArray[np.float64]]
            Random samples from the hyperexponential distribution
        """
        if size is None:
            # Choose one of the exponential components based on probs
            component = self.rng.choice(len(self.probs), p=self.probs)
            # Generate a sample from the selected exponential distribution
            return self.rng.exponential(1.0 / self.rates[component])

        # For multiple samples
        # Determine total number of samples needed
        if isinstance(size, int):
            total_samples = size
            output_shape = (size,)
        else:
            total_samples = np.prod(size)
            output_shape = size

        # Choose components for all samples
        components = self.rng.choice(len(self.probs), size=total_samples, p=self.probs)

        # Generate samples from corresponding exponential distributions
        samples = np.zeros(total_samples)
        for i, component in enumerate(components):
            samples[i] = self.rng.exponential(1.0 / self.rates[component])

        # Reshape if needed
        if isinstance(size, tuple):
            samples = samples.reshape(output_shape)

        return samples
