# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project status

This repository is an **empty scaffold** — no application code, dependencies, or tests have been written yet. `requirements.txt`, `.env.example`, and `README.md` are all placeholders with `TODO` markers tied to project phases (the README's TODOs are written in French: "à compléter en Phase X"). Every `src/` subdirectory currently contains only a `.gitkeep`. Do not assume any commands, frameworks, or conventions beyond what's described below until code actually exists — check `requirements.txt` and the relevant `src/` subdirectory before relying on a library or pattern.

There is no build, lint, or test tooling configured yet (no `pytest.ini`/`pyproject.toml`/`setup.cfg`, no CI config). When code and dependencies are added, update this file with the real commands rather than guessing.

## Intended architecture

Per the README, this is meant to become a production-style **customer segmentation and churn prediction platform** with these planned components:

- **PostgreSQL** — primary data store
- **FastAPI** (`src/api/`) — REST API serving real-time churn/segmentation predictions
- **MLflow** — experiment tracking and model registry (tracking artifacts are gitignored: `mlruns/`, `mlartifacts/`, `.mlflow/`)
- **Docker** (`docker/`) — containerization
- **Streamlit** — likely for a dashboard/UI (mentioned in README, no directory allocated yet)

Planned pipeline layout under `src/`:
- `ingestion/` — pulling in raw customer data
- `cleaning/` — data cleaning/preprocessing
- `features/` — feature engineering
- `models/` — segmentation and churn-prediction model code (trained artifacts go in the top-level `models/` dir, gitignored except `.gitkeep`)
- `api/` — FastAPI app exposing predictions

`notebooks/` is for exploratory work, `tests/` for the test suite (framework not yet chosen), `docs/` for documentation, `data/{raw,processed,external}` for datasets (all gitignored except `.gitkeep` — never commit actual data files).

## Working conventions

- Add new Python dependencies to `requirements.txt` as they're introduced (currently empty) rather than installing ad hoc.
- Add new required environment variables (PostgreSQL connection, MLflow URI, API keys, etc.) to `.env.example` as placeholders — never commit a real `.env` (already gitignored).
- The project venv lives in `venv/` (Windows-style, `venv/Scripts/`) and is gitignored.
- README section headers (Business Problem, Architecture, Dataset, Installation, Project Structure, Results, Roadmap) map to project phases — fill them in as each phase completes rather than leaving TODOs indefinitely.
