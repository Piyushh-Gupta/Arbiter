# Arbiter

A Trustworthy AI Decision Support System for Evidence-Based Claim Verification using Failure-Mode-Aware Uncertainty Routing.

## Project Purpose
Arbiter explicitly models uncertainty to route factual claims to an answer, a re-retrieval loop, or human escalation. It aims to improve decision quality by decomposing verification uncertainty into retrieval insufficiency, cross-evidence contradiction, and epistemic uncertainty.

## Architecture
The system utilizes a monolithic pipeline structured around:
- ASR processing
- Claim Extraction
- Evidence Retrieval
- NLI Verification
- Triage Routing

## Repository Structure
- `configs/`: Hydra configuration files
- `data/`: Raw and processed dataset files (ignored in version control)
- `docs/`: Project documentation and architecture specs
- `src/`: Core implementation containing the FastAPI backend, DB setup, pipeline logic, and models
- `tests/`: Automated unit and integration tests

## Setup
1. Ensure Python 3.12 and Poetry are installed.
2. Run `make install` to set up dependencies.
3. Configure your environment using `cp .env.example .env`.
