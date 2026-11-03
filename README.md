# Explainable AutoML

AutoML for tabular data with **XGBoost + Optuna** at the core and an **LLM layer** around it:
describe the task in plain words, get a validated task spec, honestly evaluated models, SHAP
explanations and a plain-language report whose every number is checked against the results.

> Work in progress. The roadmap lives in [docs/PLAN.md](docs/PLAN.md) (in Russian).

## Development

```bash
uv sync
uv run pytest
uv run ruff check .
```
