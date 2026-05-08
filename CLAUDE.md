# CLAUDE.md

This file is read automatically by Claude Code at the start of every session.
It defines how we work in this repo. Follow these instructions for all tasks.

---

## Notebooks

- All notebooks must be in **Databricks native `.py` format** — `# Databricks notebook source` header, `# COMMAND ----------` between cells, magic commands prefixed with `# MAGIC `.
- **Never create or commit `.ipynb` files.** Databricks Repos has pull issues with `.ipynb`; the native `.py` format is correct for this workflow.
- **Notebooks are thin orchestrators.** Business logic, reusable functions, and data transformations belong in `localizers/`, not in notebooks.
- New notebooks go in the repo root unless they are utility/step-wise notebooks, which go in `Notebooks/`.

## `localizers/` modules

- Write **no comments** unless the *why* is non-obvious — a hidden constraint, a workaround, a subtle invariant. Never describe what the code does; well-named identifiers do that.
- Keep functions focused on a single concern. If a function is getting long, split it.
- All configuration (language maps, game names, model constants) lives in the appropriate `*_config.py` file. Do not hardcode constants in notebooks or modules.

## README

- **Always update `README.md`** when adding a new module, a new notebook, or restructuring the repo layout. The README is the onboarding document and source of truth for repo structure.

## Commits

- Commit and push **incrementally** after each logical change — don't batch unrelated changes into one commit.
- Write commit messages that explain *why*, not just *what*.

## Architecture

```
Notebooks         → orchestration, widgets, job parameters, Databricks-specific setup only
localizers/       → all reusable logic: translation pipeline, prompts, config, QC
*_config.py       → all localization-type-specific configuration in one place
```

When in doubt: if the same logic could be useful in another notebook or game pipeline, it belongs in `localizers/`.

---

## Coming Soon (not yet enforced)

The following are backlogged and will be added to this file once scaffolding exists:

- **Type hints** — add to all functions in `localizers/` as files are touched
- **Linting** — `ruff check localizers/` before committing (needs `ruff` in `requirements.txt`)
- **Unit tests** — `tests/` directory with mocks for Spark/Databricks dependencies (needs test infrastructure)
- **`requirements.txt`** — pin all dependencies for local development
