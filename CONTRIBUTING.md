# Contributing to EquiAudit

Thank you for your interest in contributing to EquiAudit. All contributions are welcome, whether bug reports, feature suggestions, documentation improvements, or code changes.

## Getting Started

1. Fork the repository and clone your fork.
2. Create a virtual environment and install the development dependencies:

```bash
pip install -e ".[gui]"
pip install -r requirements.txt
```

3. Create a branch for your changes:

```bash
git checkout -b your-feature-name
```

## Project Structure

| Directory | Purpose |
|-----------|---------|
| `src/equiaudit/cli/` | Headless CLI and Python API entry points |
| `src/equiaudit/gui/` | Streamlit web interface (optional extra) |
| `src/equiaudit/pipeline/` | Stage definitions and pipeline orchestration |
| `src/equiaudit/tools/` | Fairness and bias-mitigation tool implementations |
| `src/equiaudit/models/` | LLM provider wrappers and agent base classes |
| `src/equiaudit/reporting/` | PDF and Markdown report generation |
| `src/equiaudit/data/` | Bundled example datasets |
| `examples/` | Example usage scripts and config |
| `docs/` | Usage and extension guides |

See [docs/EXTENDING.md](docs/EXTENDING.md) for instructions on adding new tools, agents, LLM backends, and pipeline stages.

## Making Changes

- Keep changes focused: one feature or fix per pull request.
- Follow existing code style (no comments unless the reason is non-obvious).
- Update `docs/USAGE.md` or `docs/EXTENDING.md` if the change affects user-facing behaviour.
- Test your changes with at least one dataset (e.g. `src/equiaudit/data/adult-all.csv`).

## Submitting a Pull Request

1. Push your branch to your fork.
2. Open a pull request against `master` with a short title and a description of what changed and why.
3. A maintainer will review and merge or request changes.

## Reporting Issues

Open an issue on [GitHub Issues](https://github.com/pchmelo/EquiAudit/issues) with:

- A clear description of the problem.
- Steps to reproduce it.
- The dataset and config used (redact API keys).
- The full error message or unexpected output.

## Code of Conduct

Be respectful and constructive. Contributions of all experience levels are welcome.
