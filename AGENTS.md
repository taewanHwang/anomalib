# Repository Guidelines

## Project Structure & Module Organization
- `src/anomalib/` hosts the Lightning loops, data modules, models, and CLI; keep new code in scoped subpackages and surface it via `anomalib/__init__.py`.
- Reference YAMLs live in `examples/configs/`, runnable notebooks in `examples/notebooks/`, and docs in `docs/source/`. Update all three whenever an API or CLI flag moves.
- Tests live in `tests/unit/` and `tests/integration/`; datasets, pretrained weights, and experiment outputs belong in `datasets/`, `pre_trained/`, and `results*/` respectively.

## Build, Test, and Development Commands
- `uv pip install -e ".[dev]"` (or `anomalib install --option dev`) provisions the locked toolchain from `uv.lock`.
- `pre-commit install && pre-commit run --all-files` runs Ruff, formatting, and import checks.
- `anomalib train --config examples/configs/image/patchcore.yaml --dataset.path datasets/mvtec` executes a local training job; swap the config path to match your model.
- `tox -e pre-merge-py310` mirrors CI by installing `.[full]`, running unit/integration suites with coverage, and verifying notebooks via `nbmake`.

## Coding Style & Naming Conventions
- Use Python 3.10+ features, keep type hints in public APIs, and colocate config helpers with the modules they configure.
- Ruff (see `pyproject.toml`) enforces 120-character lines, Google-style docstrings, import sorting, and selective lint rules—avoid `ruff: noqa` unless you document why.
- Stick to `snake_case` for functions/args, `PascalCase` for classes, `patchcore_<variant>.yaml` for configs, and `test_<feature>.py` naming in `tests/`.

## Testing Guidelines
- `pytest tests/unit tests/integration -v --cov=anomalib --cov-fail-under=75` is the standard gate; hit the 75% bar before opening a PR.
- Run `pytest --nbmake examples/notebooks --ignore examples/notebooks/400_openvino --ignore examples/notebooks/500_use_cases/501_dobot` whenever notebook content changes.
- For dataset-heavy tests, use `ANOMALIB_DATASET_PATH` to point fixtures at local copies and keep artifacts out of version control.

## Commit & Pull Request Guidelines
- PR titles must follow Conventional Commits (`feat(model): add dinomaly head`), and Commitizen will fail CI if they do not.
- Squash merges mean the PR title becomes the final commit; describe scope, configs touched, datasets used, and link issues in the PR body.
- Document user-facing changes in `CHANGELOG.md` plus any affected docs/notebooks before requesting review.

## Security & Configuration Tips
- Follow `SECURITY.md` for vulnerability intake and keep secrets or proprietary datasets out of Git; load them via `.env` and `python-dotenv`.
- GPU and dataset selection respect `CUDA_VISIBLE_DEVICES` and `ANOMALIB_DATASET_PATH`; set them locally and inside `tox` for reproducible runs.
- Use the vetted automation in `tools/` or tox targets (`snyk-scan`, `bandit-scan`, `trivy-scan`) instead of ad-hoc scripts when scanning dependencies or exports.
