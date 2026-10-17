# Readable Python before experiments

The active `experiments/worm_uncertainty/*.py` scripts use Ruff 0.14.1 to
standardize four-space indentation, quotes, line wrapping and multiline statements.
Basic lint checks catch syntax problems and undefined names without importing
CUDA libraries or running experiments. `.editorconfig` supplies shared editor
indentation settings. Historical experiments, inherited upstream code and
submodules are outside the formatting scope.

Install the small development tool in its own environment from the repository
root; this keeps it separate from the research environment:

```bash
python3 -m venv .venv-style
.venv-style/bin/python -m pip install -r requirements-dev.txt
make format-diff PYTHON=.venv-style/bin/python  # preview changes
make format PYTHON=.venv-style/bin/python       # apply formatting
make check PYTHON=.venv-style/bin/python        # check before running an experiment
```

These commands run on the CPU and do not launch training. A failed `make check`
returns a nonzero exit code. To enforce the check before a command, join them with
`&&`, for example (with the research environment already activated):

```bash
make check PYTHON=.venv-style/bin/python && \
  python -B experiments/worm_uncertainty/diagnose_3d.py \
    --output output/worm_uncertainty/known_3d_checked_repeat
```

Choose a fresh output directory as usual. Existing experiment scripts remain
usable directly; the check is explicit. The GitHub workflow runs the same style
check on pushes and pull requests affecting these files. Formatting does not
replace scientific tests or convergence checks.

For VS Code, install the Ruff extension (`charliermarsh.ruff`), select it as the
Python default formatter and enable **Editor: Format On Save**. It reads the
repository's `pyproject.toml`. Other editors can invoke `ruff format` on save.

The initial cleanup checked equivalent Python ASTs (normalizing split import
statements) and passed all 24 worm CPU tests. Recorded experiment reports keep
their original code hashes: formatting changes current source bytes, so hashes
in historical reports identify the source at the original run, not the formatted
working tree. Do not rewrite historical hashes to match current code.
