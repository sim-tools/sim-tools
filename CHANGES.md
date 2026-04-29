# Change log

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html). Dates formatted as YYYY-MM-DD as per [ISO standard](https://www.iso.org/iso-8601-date-and-time-format.html).

Consistent identifier (represents all versions, resolves to latest): [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.4553641.svg)](https://doi.org/10.5281/zenodo.4553641)

## v1.1.0

Add support for nested dictionaries in `DistributionRegistry`.

### Added

* `DistributionRegistry` now accepts dictionaries from a nested configuration, and can return these in the nested structure if set `preserve_structure=True`.

## v1.0.4

Fix to `NSPPThinning` sampling approach. 

### Fixed

* `NSPPThinning.sample()` fixed so that `t` falls into the the time interval where the candidate arrival will occur. This allows the correct acceptance probability to be used.  Prior to this fix t was occuring in the *current* time interval.

### Changed

* **NSPPThinning** now pre-computes the acceptance probabilities on initialisation.  This is means that there are no repeated calculations (divisions) during a simulation run and lookup via `numpy` rather than `pandas`.
* **NSPPThinning** now validates the `data` parameter: is it a `DataFrame (`TypeError`), does it contain the correct columns (`ValueError`), and is it empty/have more than 1 row (`ValueEror`) .

## v1.0.3

### Changed

* Switched to trusted publishing for PyPI
* Migrated repository from tommonks personal to sim-tools organisation
* Updated package URLS.

## v1.0.2

### Fixed

*   **Publish ot PyPI**: Github action to publish to pypi modified to use release/v1 (prior version was sunsetted).

## v1.0.1

### Fixed

* **Type Hinting and Pylance Support**: Adjusted type hinting within `@register` decorator to enable display of docstrings, hover hints and argument suggestions.

## v1.0.0

v1.0.0 centres on **multi-metric support** - `confidence_interval_method` and `ReplicationsAlgorithm` can now analyse multiple metrics at once, driving changes to syntax (explicit `metrics` argument), adapter formats, defaults, docs, plotting, and tests to support the new capability.

### Added

* **Multi-metric support:** Both `confidence_interval_method` and `ReplicationsAlgorithm` can now process multiple metrics at once. Algorithm now requires explicit `metrics argument` and new adapter format (⚠ breaking change).
* **Plot improvement:** `plotly_confidence_interval_method` now uses shaded confidence intervals (dashed still available).
* **Documentation:** Add a dedicated page on `confidence_interval_method`.
* **Testing:** New tests added following multiple metrics changes.

### Changed

* **Defaults:** The default for `desired_precision` in `confidence_interval_method` is now 0.1 (was 0.05) for consistency with `ReplicationsAlgorithm`.
* **Protocols:** Updated `ReplicationsAlgorithmModelAdapter` and add `AlgorithmObserver` to accommodate the new required format (more complex than `ReplicationObserver`).
* **Documentation:** Updated the algorithm documentation for the new syntax (e.g. new treat-sim adapter), and moved `treat-sim` description to its own page to avoid repetition.
* **Docstrings:** Improvements in `output_analysis.py`.
* **Testing:** Amended to work with new syntax/logic of replications methods.
* **Linting:** Linting several files.

### Fixed

* **Algorithm:** Now adjusts result if solution was found within initial replications.

###

## [v0.10.0](https://github.com/TomMonks/sim-tools/releases/tag/v0.10.0)[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.16754108.svg)](https://doi.org/10.5281/zenodo.16754108)

### Added

* Add `sort` argument to `DistributionRegistry.create_batch()` for dict inputs, ensuring deterministic results if the config key order changes.
* Add `_validate_and_create()` to `DistributionRegistry` which checks that the distribution configurations are dictionaries with only two keys: "class_name" and "params".
* Add tests for the new `sort` argument and `_validate_and_create()` method.

### Changed

* Some linting of `_validation.py`, `time_dependent.py`, `distributions.py` and `trace.py`.

## [v0.9.1](https://github.com/TomMonks/sim-tools/releases/tag/v0.9.1)[[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.16754108.svg)](https://doi.org/10.5281/zenodo.16754108)]

### Added

* Add unit tests for all distributions which check it uses the base class, data types and sample size are correct, that the sample mean looks right, and that the random seed is working. Some similar tests existed, but they did not cover all distributions.
* Add some specific tests for `Lognormal` and `DiscreteEmpirical`.
* Add back tests for all distributions (which check that new samples are equal to those generated previously, when random seed controlled).

### Changed

* Adjusted docstrings to use a more consistent NumPy style.
* Some linting of `distributions.py`.

### Fixed

* `DiscreteEmpirical` - now allows any type of data to be included. For example, str, as well as numeric value.

## [v0.9.0](https://github.com/TomMonks/sim-tools/releases/tag/v0.9.0)  [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15256118.svg)](https://doi.org/10.5281/zenodo.15256118)

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

## [v0.8.0](https://github.com/TomMonks/sim-tools/releases/tag/v0.8.0a) [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15041282.svg)](https://doi.org/10.5281/zenodo.15041282)

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

### Changed

* Update Github action to publish to pypi. Use setuptools instead of build

## [v0.3.1](https://github.com/TomMonks/sim-tools/releases/tag/v0.3.1) [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.10625470.svg)](https://doi.org/10.5281/zenodo.10625470)

### Fixed

* PYPI has deprecated username and password. PYPI Publish Github action no works with API Token

## [v0.3.0](https://github.com/TomMonks/sim-tools/releases/tag/v0.3.0) [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.10625096.svg)](https://doi.org/10.5281/zenodo.10625096)

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

## [v0.2.1](https://github.com/TomMonks/sim-tools/releases/tag/v0.2.1) [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.10201794.svg)](https://doi.org/10.5281/zenodo.10201794)

### Fixed

* Modified Setup tools to avoid numpy import error on build.
* Updated github action to use up to date actions.

## [v0.2.0](https://github.com/TomMonks/sim-tools/releases/tag/v0.2.0) [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.10201726.svg)](https://doi.org/10.5281/zenodo.10201726)

### Added

* Added `sim_tools.distribution` module.  This contains classes representing popular sampling distributions for Discrete-event simulation. All classes encapsulate a `numpy.random.Generator` object, a random seed, and the parameters of a sampling distribution.

### Changed

* Python has been updated, tested, and patched for 3.10 and 3.11 as well as numpy 1.20+
* Minor linting and code formatting improvement.
