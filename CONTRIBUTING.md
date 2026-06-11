# Contributing

We welcome contributions to `sim-tools`.

## Feature requests and bug reports

Before opening an issue, please search existing issues to avoid duplicates. If an issue exists, you can add a comment with additional details and/or upvote (👍) the issue. If there is not an existing issue, please open one and provide as much detail as possible.

## Code contributions

You are welcome to make contributions to the code. To do so please:

1. Fork the repository.
2. Create a branch for your feature or fix.
3. Make changes then commit them with clear, descriptive messages.
4. Open a pull request against our repository. Describe your changes and reference any related GitHub issues.

## Environment

You can install development dependencies from `pyproject.toml` using your preferred environment manager. For example:

```bash
conda create -n sim_tools python=3.12
conda activate sim_tools
pip install -e .
pip install --group dev
```

A pinned conda environment environment is also provided:

```bash
conda env create -f binder/environment.yml
conda activate sim_tools
```

## Tests

```bash
pytest
```

## Documentation

The documentation is created with `great-docs`.

Generate and render the site:

```bash
great-docs build
```

Open in your browser:

```bash
great-docs preview
```

## Style

We use Ruff, with the configuration defined in `pyproject.toml`. To lint and format the repository, run:

```bash
ruff format
ruff check --fix .
```

NumPy style docstrings are used.
