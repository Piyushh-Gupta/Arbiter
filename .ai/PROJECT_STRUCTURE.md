# Project Structure

## Directory Responsibilities
- `.ai/`: Permanent AI Governance Layer. Reference documents for AI workflows.
- `.github/`: CI/CD automation pipelines.
- `configs/`: Hydra YAML configurations.
- `data/`: Raw and processed dataset files, plus FAISS/BM25 indices. (Git ignored).
- `docs/`: Original SRD, SDD, and MEEP documentation.
- `scripts/`: Executable entrypoints for dataset construction and training pipelines.
- `src/`: Core Python application codebase (FastAPI, Database, Models, Pipeline).
- `tests/`: Pytest suite (Unit, Integration, Fixtures).
