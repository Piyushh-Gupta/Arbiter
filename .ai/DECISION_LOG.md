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
| Architecture | ADR-005: Immutable Dataset Registry | All dataset metadata must remain immutable and be managed exclusively through the Dataset Registry to guarantee deterministic behavior and reproducible experiments. | Approved |
| Governance | ADR-006: CI Contract | Establish `.ai/CI_CONTRACT.md` as the single source of truth for deterministic local and GitHub Actions validation. | Approved |
| Architecture | ADR-007: Transport Metadata Separation | Dataset metadata and transport metadata are separate architectural concerns and must evolve independently. | Approved |
| Architecture | ADR-008: Pure Validation Layer | Dataset validation is a read-only evaluation process that always returns a deterministic validation report. Exception handling and orchestration remain outside the validator. | Approved |
| Architecture | ADR-009: Stateless Artifact Resolution | ArtifactManager derives lifecycle from observable filesystem state. Validation sentinel files are performance caches written exclusively by the Validator and never represent mutable application state. | Approved |
| Architecture | ADR-010: Documentation Ownership | Dataset manifests are the exclusive source of documentation metadata. The Dataset Registry owns operational metadata only. Manifest loading remains stateless and deterministic through ProjectPaths. | Approved |
