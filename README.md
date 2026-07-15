# Arbiter

**A Trustworthy AI Decision Support System for Evidence-Based Claim Verification using Failure-Mode-Aware Uncertainty Routing.**

## Project Overview

Arbiter is an evidence-aware decision support system designed to evaluate factual claims and explicitly reason about verification uncertainty. Instead of a single confidence score, Arbiter models uncertainty across three failure modes:
1. Retrieval insufficiency
2. Cross-evidence contradiction
3. Model epistemic uncertainty

It uses this decomposition to intelligently route claims to an answer, a re-retrieval loop, or human escalation.

## Architecture Summary

The monolithic pipeline consists of five deterministic stages:
1. **ASR**: Whisper-based transcription of spoken claims.
2. **Claim Extraction**: DistilBERT-based binary classification to find verifiable assertions.
3. **Retrieval**: FAISS + BM25 hybrid search over a static FEVER Wikipedia corpus snapshot.
4. **Verification**: DeBERTa-v3 based NLI entailment with MC-Dropout uncertainty estimation.
5. **Triage Head**: XGBoost routing policy over the three decomposed uncertainty signals.

## Tech Stack

- **Python**: 3.12
- **Backend API**: FastAPI, Uvicorn
- **Database**: SQLite, SQLAlchemy
- **Configuration**: Hydra, Pydantic Settings
- **Machine Learning**: PyTorch, Transformers, XGBoost, FAISS
- **Experiment Tracking**: Weights & Biases (W&B)
- **Dependency Management**: Poetry
- **CI/CD & Containerization**: GitHub Actions, Docker, Docker Compose

## Repository Structure

```text
Arbiter/
├── .github/          # GitHub Actions CI/CD workflows
├── configs/          # Hydra configuration yaml files
├── data/             # Raw, processed data and indices (ignored in git)
├── docs/             # SDD, SRD, MEEP and other documentation
├── notebooks/        # Jupyter notebooks for EDA and prototyping
├── scripts/          # Offline dataset generation and training scripts
├── src/              # Core application and model code
│   ├── api/          # FastAPI routes, schemas, and entrypoint
│   ├── core/         # Settings, constants, exceptions, logging
│   ├── database/     # SQLAlchemy sessions and models
│   ├── evaluation/   # Risk-coverage metric calculations
│   ├── models/       # PyTorch and XGBoost wrappers
│   ├── pipeline/     # End-to-end inference orchestrator
│   ├── training/     # Training loop utilities
│   └── utils/        # Shared helpers
└── tests/            # Unit and integration tests
```

## Getting Started

### Prerequisites
- Python 3.12+
- Poetry 1.8+
- Docker and Docker Compose (optional for deployment)

### Installation
1. Install dependencies:
   ```bash
   make install
   ```
2. Set up your environment:
   ```bash
   cp .env.example .env
   ```

### Development
- **Run the API server**: `make run`
- **Format code**: `make format`
- **Lint code**: `make lint`
- **Typecheck**: `make typecheck`
- **Run tests**: `make test`

### Docker Deployment
Build and run the API using Docker Compose:
```bash
make docker-build
make docker
```
