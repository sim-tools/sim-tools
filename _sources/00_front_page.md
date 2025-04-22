```{image} ./img/simtools_logo_purple.png
:alt: simtools
:width: 400px
:align: center
```


<p align="center">
  <i align="center">Tools to support Discrete-Event Simulation (DES) and Monte-Carlo Simulation education and practice</i>
</p>

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/TomMonks/sim-tools/HEAD)
[![DOI](https://zenodo.org/badge/225608065.svg)](https://zenodo.org/badge/latestdoi/225608065)
[![PyPI version fury.io](https://badge.fury.io/py/sim-tools.svg)](https://pypi.python.org/pypi/sim-tools/)
[![Anaconda-Server Badge](https://anaconda.org/conda-forge/sim-tools/badges/version.svg)](https://anaconda.org/conda-forge/sim-tools)
[![Anaconda-Server Badge](https://anaconda.org/conda-forge/sim-tools/badges/platforms.svg)](https://anaconda.org/conda-forge/sim-tools)
[![Read the Docs](https://readthedocs.org/projects/pip/badge/?version=latest)](https://tommonks.github.io/sim-tools)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/release/python-360+/)
[![License: MIT](https://img.shields.io/badge/ORCID-0000--0003--2631--4481-brightgreen)](https://orcid.org/0000-0003-2631-4481)


`sim-tools` is being developed to support Discrete-Event Simulation (DES) and Monte-Carlo Simulation education and applied simulation research.  It is MIT licensed and freely available to practitioners, students and researchers via [PyPi](https://pypi.org/project/sim-tools/) and [conda-forge](https://anaconda.org/conda-forge/sim-tools)

 # Vision for sim-tools

 1. Deliver high quality reliable code for DES and Monte-Carlo Simulation education and practice with full documentation.
 2. Provide a simple to use pythonic interface.
 3. To improve the quality of simulation education using FOSS tools and encourage the use of best practice.

# Features:

1. Implementation of classic Optimisation via Simulation procedures such as KN, KN++, OBCA and OBCA-m
2. Theoretical and empirical distributions module that includes classes that encapsulate a random number stream, seed, and distribution parameters.
3. An extendable Distribution registry that provides a quick reproduible way to parameterise simulation models.
4. Implementation of Thinning to sample from Non-stationary Poisson Processes (time-dependent) in a DES.
5. Automatic selection of the number of replications to run via the Replications Algorithm.
6. EXPERIMENTAL: model trace functionality to support debugging of simulation models.

## Installation

### Pip and PyPi

```bash
pip install sim-tools
```

### Conda-forge

```bash
conda install -c conda-forge sim-tools
```

### Mamba

`mamba` is a FOSS alternative to `conda` that is also quicker at resolving and installing environments.

```bash
mamba install sim-tools
```

### Binder

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/TomMonks/sim-tools/HEAD)


## Learn how to use `sim-tools`

* Online documentation: https://tommonks.github.io/sim-tools
* Introduction to DES in python: https://health-data-science-or.github.io/simpy-streamlit-tutorial/

## Citation

If you use sim0tools for research, a practical report, education or any reason please include the following citation.

> Monks, Thomas. (2021). sim-tools: tools to support the forecasting process in python. Zenodo. http://doi.org/10.5281/zenodo.4553642

```tex
@software{sim_tools,
  author       = {Thomas Monks},
  title        = {sim-tools: fundamental tools to support the simulation process in python},
  year         = {2021},
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.4553642},
  url          = {http://doi.org/10.5281/zenodo.4553642}
}
```

# Online Tutorials

* Optimisation Via Simulation [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/TomMonks/sim-tools/blob/master/docs/02_ovs/03_sw21_tutorial.ipynb)


## Contributing to sim-tools

Please fork Dev, make your modifications, run the unit tests and submit a pull request for review.

Development environment:

* `conda env create -f binder/environment.yml`

* `conda activate sim_tools`

**All contributions are welcome!**

## Tips

Once in the `sim_tools` environment, you can run tests using the following command:

```
pytest
```

To view the documentation, navigate to the top level directory of the code repository in your terminal and issue the following command to build the Jupyter Book:

```
jb build docs/
```

To lint the repository, run:

```
bash lint.sh
```
