# Arbiter

A Trustworthy AI Decision Support System for Evidence-Based Claim Verification using Failure-Mode-Aware Uncertainty Routing.

## Project Purpose
Arbiter is designed to explicitly model uncertainty to route factual claims to an answer, a re-retrieval loop, or human escalation. It aims to improve decision quality by decomposing verification uncertainty into retrieval insufficiency, cross-evidence contradiction, and epistemic uncertainty.

## Planned Architecture
*Note: The machine learning pipeline is currently in the design and infrastructure setup phase and has not yet been implemented.*

Once implemented, the system will utilize a monolithic pipeline structured around:
- ASR processing (Planned)
- Claim Extraction (Planned)
- Evidence Retrieval (Planned)
- NLI Verification (Planned)
- Triage Routing (Planned)

## Current Repository Structure
The foundational repository infrastructure has been established:
- `configs/`: Hydra configuration files
- `data/`: Raw and processed dataset files (ignored in version control)
- `docs/`: Project documentation and architecture specs
- `src/`: Core implementation containing the FastAPI backend bootstrap and database setup
- `tests/`: Automated unit and integration tests

## Setup
1. Ensure Python 3.12 and Poetry are installed.
2. Run `make install` to set up dependencies.
3. Configure your environment using `cp .env.example .env`.
