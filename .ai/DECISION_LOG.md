# Decision Log

| Date | Decision | Rationale | Status |
|------|----------|-----------|--------|
| Project Inception | Frozen Architecture | Enforce reproducibility and scope boundaries via SRD/SDD/MEEP. | Approved |
| Infrastructure | Poetry package-mode=false | Arbiter is a deployed service, not a PyPI package. | Approved |
| Infrastructure | XGBoost for Triage | Built-in feature importance simplifies required ablation studies. | Approved |
| Infrastructure | Local FAISS vs Vector DB | Avoids networking overhead and docker-compose complexity for static datasets. | Approved |
| Governance | `.ai/` Governance Layer | Maintains AI assistant alignment over long development lifecycles. | Approved |
| Infrastructure | CI Stabilization & Strict Typing | Enforce zero-global-state testing and mypy-strict compliance baseline. | Approved |
| Infrastructure | ADR-002: Track `poetry.lock` | Track `poetry.lock` for deterministic builds. | Approved |
| Governance | ADR-003: Milestone IDs | Adopt permanent milestone IDs as the project standard. | Approved |
| Architecture | ADR-004: Bootstrap inside Lifespan | Application startup must execute the bootstrap layer through the FastAPI lifespan to guarantee deterministic initialization, validation, and fail-fast startup. | Approved |
