# M4.4 Risk & Failure-Aware Decision Engine - Tasks

- [x] Add risk policy exceptions to `src/core/exceptions.py`
- [x] Define protocol `BaseRiskPolicy` in `src/core/decision/base.py`
- [x] Implement `RiskTrace`, `RiskEvaluation`, and `RiskPolicyRegistry` in `src/core/decision/decision_models.py`
- [x] Implement concrete policies in `src/core/decision/policies.py`
- [x] Update `DecisionPolicyEngine` and `PolicyDecisionStrategy` in `src/core/decision/implementations.py`
- [x] Re-export entities in `src/core/decision/__init__.py`
- [x] Update `build_decision_registry` in `src/core/bootstrap.py`
- [x] Create unit and integration tests in `tests/unit/test_decision_risk_policies.py`
- [x] Update `.ai/MILESTONE_STATUS.md` and `.ai/DECISION_LOG.md`
- [x] Run full validation suite (ruff, isort, mypy, pytest with coverage)
- [x] Execute git workflow (stage, commit, push, check CI status)
