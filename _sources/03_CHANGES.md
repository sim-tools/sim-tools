# Change log

## [v0.9.0]

### Added

* `distributions.DistributionRegistry` - for batch creation of standard distributions from a dictionary or list.  
* DOCS: `DistributionRegistry` explaination and examples including use of JSON to store configs.
* `distributions.spawn_seeds` function to support creation of PRNG streams.
* `Hyperexponential` - A continuous probability distribution that is a mixture (weighted sum) of
    exponential distributions. It has a higher coefficient of variation than
    a single exponential distribution, making it useful for modeling highly
    variable processes or heavy-tailed phenomena.
* `RawContinuousEmpirical` - A distribution that performs linear interpolation between data points according to 
    the algorithm described in Law & Kelton's "Simulation Modeling and Analysis".
* `sim_tools._validation`: internal module that contains common validation routines for `sim_tools` functions and classes.
* All distribution classes updated to include valudation of input parameters.
* DOCS: Dedicated page for using empirical distributions.


### Changed

* `Distribution` changed from abstract base class to `Protocol`.  All inheritance removed from concrete classes. 
* Added `__repr__()` to all distribution classes.
* DOCS: improved docstrings for all distribution classes
* BREAKING: `Discrete` -> `DiscreteEmpirical`
* BREAKING: `RawEmpirical` -> `RawDiscreteEmpirical`
* BREAKING: `ContinuousEmpirical` -> GroupedContinuousEmpirical`. To clarify the purpose of the emprical distribution
* BREAKING: `NSPPThinning`: class now only requires "mean_iat" column in `data`. Acceptance/rejection calcualted using $iat_{min} / iat_(t)$


### Fixed

* `Gamma` fix of the calculation of the mean based on alpha and beta.
* `GroupedContinuousEmpirical`: silent bug when `u` selects the first group.  Interpolation did not work correctly and sampled out of range. This now been handled by logic pre-sample.
* `NSPPThinning`: removed redundant outer loop from sampling.

## [v0.8.0](https://github.com/TomMonks/sim-tools/releases/tag/v0.8.0a)

### Added

* Add `simpy` and `treat-sim` to the environment, as these were required in the notebooks in `docs/`.
* Add `nbqa` and `pylint` to the environment for linting, plus a relevant files `lint.sh` and `.pylintrc`.
* Add tests for `output_analysis` functions (functional, unit and back tests).
* Add validation of parameters in `ReplicationsAlgorithm`.
* Add validation of data type in `OnlineStatistics`.

### Changed

* Simplified distribution value type tests to a single test where possible using `pytest.mark.parametrize`.
* Linted `.py` and `.ipynb` files using `pylint` (most addressed, some remain unresolved).
* Provided advice on tests, building docs and linting in the `README.md`.
* `00_front_page.md` now just imported `README.md` (reducing duplication, and keping it up-to-date).

### Removed

* Removed duplicate `sw21_tutorial.ipynb`.

### Fixed

* Within `confidence_interval_method`, convert data provided to `OnlineStatistics` to `np.array` so that it is actually used to update the class (when before, it was not, as it was a list).

## [v0.7.1](https://github.com/TomMonks/sim-tools/releases/tag/v0.7.1)  [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.14844701.svg)](https://doi.org/10.5281/zenodo.14844701)

### Fixed

* Patched `ReplicationsAlgorithm` look ahead will now correctly use `_klimit()` to calculate extra no. replications to run.

## [v0.7.0](https://github.com/TomMonks/sim-tools/releases/tag/v0.7.0) [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.14834956.svg)](https://doi.org/10.5281/zenodo.14834956) 

### Added

* `output_analysis` module - focussed at the moment on selecting the number of replications
* `ReplicationsAlgorithm` that implements the automated approach to selecting the number of replications for a single performance measures.
* `ReplicationsAlgorithmModelAdapter` - a `Protocol` to adapt any model to work with with `ReplicationsAlgorithm`
* `confidence_interval_method` - select the number of replication using the classical confidence interval method
* `plotly_confidence_interval_method` - visualise the confidence interval method using plotly.
* `ReplicationObserver` a `Protocol` for observering the replications algorithm 
*  `ReplicationTabulizer` record replications algorithm in a pandas dataframe. 
* Documentation for `ReplicationsAlgorithm`

### Updated

* `sim-tools` dev conda environment now pip installs local python package in editable model.

## [v0.6.1](https://github.com/TomMonks/sim-tools/releases/tag/v0.6.1) [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.13135700.svg)](https://doi.org/10.5281/zenodo.13135700)

### Fixed

* BUILD: added rich library. 

### Removed

* Scipy Dependency

## [v0.6.0](https://github.com/TomMonks/sim-tools/releases/tag/v0.6.0) [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.13122484.svg)](https://doi.org/10.5281/zenodo.13122484)

### Added

* Added `nspp_plot` and `nspp_simulation` functions to `time_dependent` module.
* DOCS: added `nspp_plot` and `nspp_simulation` examples to time dependent notebook
* DOCS: simple trace notebook

### Changed

* BREAKING: to prototype trace functionality. config name -> class breaks with v0.5.0

### Fixed

* THINNING: patched compatibility of thinning algorithm to work with numpy >= v2. `np.Inf` -> `np.inf`

## [v0.5.0](https://github.com/TomMonks/sim-tools/releases/tag/v0.5.0)  [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.12204481.svg)](https://doi.org/10.5281/zenodo.12204481)

### Added

* EXPERIMENTAL: added `trace` module with `Traceable` class for colour coding output from different processes and tracking individual patients.

### Fixed

* DIST: fix to `NSPPThinning` sampling to pre-calcualte mean IAT to ensure that correct exponential mean is used.
* DIST: normal distribution allows minimum value and truncates automaticalled instead of resampling.

## [v0.4.0](https://github.com/TomMonks/sim-tools/releases/tag/v0.4.0) [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.10987685.svg)](https://doi.org/10.5281/zenodo.10987685)

### Changed

* BUILD: Dropped legacy `setuptools` and migrated package build to `hatch`
* BUILD: Removed `setup.py`, `requirements.txt` and `MANIFEST` in favour of `pyproject.toml`

## [v0.3.3](https://github.com/TomMonks/sim-tools/releases/tag/v0.3.3) [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.10629861.svg)](https://doi.org/10.5281/zenodo.10629861)

### Fixed

* PATCH: `distributions.Discrete` was not returning numpy arrays.

## [v0.3.2](https://github.com/TomMonks/sim-tools/releases/tag/v0.3.2) [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.10625581.svg)](https://doi.org/10.5281/zenodo.10625581)

## Changed 

* Update Github action to publish to pypi. Use setuptools instead of build

## [v0.3.1](https://github.com/TomMonks/sim-tools/releases/tag/v0.3.1) [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.10625470.svg)](https://doi.org/10.5281/zenodo.10625470)

### Fixed:

* PYPI has deprecated username and password. PYPI Publish Github action no works with API Token

## [v0.3.0](https://github.com/TomMonks/sim-tools/releases/tag/v0.3.0)

### Added

* Distributions classes now have python type hints.
* Added distributions and time dependent arrivals via thinning example notebooks.
* Added `datasets` module and function to load example NSPP dataset.
* Distributions added
    * Erlang (mean and stdev parameters)
    * ErlangK (k and theta parameters)
    * Poisson
    * Beta
    * Gamma
    * Weibull
    * PearsonV
    * PearsonVI
    * Discrete (values and observed frequency parameters)
    * ContinuousEmpirical (linear interpolation between groups)
    * RawEmpirical (resample with replacement from individual X's)
    * TruncatedDistribution (arbitrary truncation of any distribution)
* Added sim_tools.time_dependent module that contains `NSPPThinning` class for modelling time dependent arrival processes.
* Updated test suite for distributions and thinning
* Basic Jupyterbook of documentation.

## [v0.2.1](https://github.com/TomMonks/sim-tools/releases/tag/v0.2.1)

### Fixed

* Modified Setup tools to avoid numpy import error on build.
* Updated github action to use up to date actions.

## v0.2.0 [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.10201726.svg)](https://doi.org/10.5281/zenodo.10201726)

### Added

* Added `sim_tools.distribution` module.  This contains classes representing popular sampling distributions for Discrete-event simulation. All classes encapsulate a `numpy.random.Generator` object, a random seed, and the parameters of a sampling distribution.  

### Changed

* Python has been updated, tested, and patched for 3.10 and 3.11 as well as numpy 1.20+
* Minor linting and code formatting improvement.