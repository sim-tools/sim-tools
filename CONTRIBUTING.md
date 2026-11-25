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

A pinned `conda` environment is also provided:

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

We use Ruff to the lint Python files, with the configuration defined in `pyproject.toml`. To lint and format the repository, run:

```bash
ruff format
ruff check --fix .
```

To lint Python code in the Quarto files, we use `lintquarto`, with configuration likewise in `pyproject.toml`. To run:

```bash
lintquarto
```

NumPy style docstrings are used.

## Spellcheck

There is a spellcheck GitHub action.

If you want to run the spellcheck locally, you need to install vale. On Linux:

```
sudo apt update
sudo apt install snapd
sudo snap install vale
```

With vale installed, run:

```
vale sync
vale .
```

## Contributors

If your name or contributions are missing from the README, or if you contributed in ways not captured by the current role emojis, please create an issue and use:

```
@all-contributors please add @githubuser for ...
```

Then list appropriate contribution types from [allcontributors.org/docs/en/emoji-key](https://allcontributors.org/docs/en/emoji-key) (e.g., code, review, doc, content, bug, ideas, infra).

Alternatively, you can update it from the command line. This may be preferable, as the bot will create GitHub issues that email people when they are added.

You'll need to install the [All-Contributors CLI tool](https://allcontributors.org/cli/installation/):

```
npm i -D all-contributors-cli
```

You can then run the following and select/enter relevant information when prompted:

```
npx all-contributors
```

If you want to remove specific contributions or people, edit the `.all-contributorsrc` file then run the following to regenerate the table in `README.md`. (Don't edit `README.md`, as it is just generated based on `.all-contributorsrc`).

```
npx all-contributors generate
```