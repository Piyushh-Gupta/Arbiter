"""Immutable domain models for Decision Engine Architecture Modernization (M4.1 & M4.2)."""

import hashlib
import json
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, model_validator


class DecisionAction(str, Enum):
    """Closed vocabulary of final decision actions."""

    ACCEPT = "ACCEPT"
    REJECT = "REJECT"
    ESCALATE = "ESCALATE"
    ABSTAIN = "ABSTAIN"


class DecisionDefinition(BaseModel):
    """Immutable configuration for decision engine policy strategies."""

    decision_strategy: str = Field(default="policy")
    confidence_policy: str = Field(default="calibrated")
    uncertainty_policy: str = Field(default="threshold_based")
    failure_policy: str = Field(default="severity_aware")
    escalation_policy: str = Field(default="default")

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)


class ThresholdDecisionDefinition(DecisionDefinition):
    """Configuration for threshold-based decision routing (M12 backward compatibility)."""

    accept_max_uncertainty: float = Field(default=0.3, ge=0.0, le=1.0)
    reject_max_uncertainty: float = Field(default=0.7, ge=0.0, le=1.0)

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)


class DecisionRule(BaseModel):
    """Immutable decision policy rule mapping conditions to a routing action."""

    rule_id: str = Field(..., min_length=1)
    priority: int = Field(default=1, ge=1)
    enabled: bool = Field(default=True)
    conditions: dict[str, Any] = Field(default_factory=dict)
    action: str = Field(..., min_length=1)

    model_config = ConfigDict(frozen=True)


class DecisionPolicyGroup(BaseModel):
    """Immutable collection of related decision rules with group-level priority and enablement."""

    group_id: str = Field(..., min_length=1)
    priority: int = Field(default=1, ge=1)
    enabled: bool = Field(default=True)
    ordered_rules: tuple[DecisionRule, ...] = Field(default_factory=tuple)

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)


class DecisionPolicyResult(BaseModel):
    """Immutable outcome of evaluating a single DecisionPolicyGroup."""

    group_id: str = Field(..., min_length=1)
    matched_rules: tuple[str, ...] = Field(default_factory=tuple)
    confidence_delta: float = Field(default=0.0)
    uncertainty_delta: float = Field(default=0.0)
    escalation_requested: bool = Field(default=False)
    selected_action: str | None = Field(default=None)
    reasoning: tuple[str, ...] = Field(default_factory=tuple)

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)


class DecisionRuleEvaluation(BaseModel):
    """Immutable record of evaluating an individual DecisionRule."""

    rule_id: str = Field(..., min_length=1)
    matched: bool = Field(...)
    confidence_delta: float = Field(default=0.0)
    uncertainty_delta: float = Field(default=0.0)
    explanation: str = Field(default="")
    priority: int = Field(default=1, ge=1)

    model_config = ConfigDict(frozen=True)


class DecisionRuntimeMetadata(BaseModel):
    """Immutable execution environment metadata for policy engine evaluation."""

    policy_engine: str = Field(..., min_length=1)
    configuration_fingerprint: str = Field(..., min_length=1)
    schema_version: str = Field(default="1.0")
    execution_timestamp: str = Field(default="2026-08-01T00:00:00Z")
    execution_environment: str = Field(default="production")

    model_config = ConfigDict(frozen=True)


class DecisionExecutionMetadata(BaseModel):
    """Immutable operational provenance for decision execution."""

    request_id: str = Field(..., min_length=1)
    execution_duration_ms: float = Field(default=0.0, ge=0.0)
    profile: str = Field(..., min_length=1)
    decision_policy: str = Field(..., min_length=1)

    model_config = ConfigDict(frozen=True)


class DecisionEngineMetadata(BaseModel):
    """Immutable structural engine metadata capturing execution complexity and fingerprints."""

    engine_version: str = Field(default="1.0")
    policy_engine_version: str = Field(default="1.0")
    evaluated_group_count: int = Field(default=0, ge=0)
    evaluated_rule_count: int = Field(default=0, ge=0)
    configuration_fingerprint: str = Field(default="")
    execution_fingerprint: str = Field(default="")

    model_config = ConfigDict(frozen=True)


class DecisionContext(BaseModel):
    """Immutable evaluation context encapsulating upstream pipeline outputs."""

    evidence_bundle: Any | None = Field(default=None)
    verification_result: Any | None = Field(default=None)
    calibration_result: Any | None = Field(default=None)
    failure_analysis_result: Any | None = Field(default=None)
    root_cause_result: Any | None = Field(default=None)
    severity_result: Any | None = Field(default=None)
    explanation_result: Any | None = Field(default=None)
    metadata: dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)


class DecisionInput(BaseModel):
    """Immutable payload binding a DecisionContext with its DecisionDefinition."""

    context: DecisionContext = Field(...)
    definition: DecisionDefinition = Field(...)

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)


class DecisionExecutionContext(BaseModel):
    """Immutable context encapsulating the complete execution outcome of policy evaluation."""

    ordered_policy_results: tuple[DecisionPolicyResult, ...] = Field(
        default_factory=tuple
    )
    ordered_rule_evaluations: tuple[DecisionRuleEvaluation, ...] = Field(
        default_factory=tuple
    )
    runtime_metadata: DecisionRuntimeMetadata = Field(...)
    execution_metadata: DecisionExecutionMetadata = Field(...)
    engine_metadata: DecisionEngineMetadata = Field(...)
    selected_action: str = Field(default="ABSTAIN")

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)


class DecisionTrace(BaseModel):
    """Immutable audit trace of decision rule evaluation and policy path execution."""

    evaluated_rules: tuple[str, ...] = Field(default_factory=tuple)
    rejected_rules: tuple[str, ...] = Field(default_factory=tuple)
    selected_rule: str | None = Field(default=None)
    confidence_evolution: tuple[float, ...] = Field(default_factory=tuple)
    uncertainty_evolution: tuple[float, ...] = Field(default_factory=tuple)
    escalation_reasoning: tuple[str, ...] = Field(default_factory=tuple)
    policy_path: tuple[str, ...] = Field(default_factory=tuple)

    model_config = ConfigDict(frozen=True)


class DecisionMetadata(BaseModel):
    """Immutable execution metadata for generated decision results."""

    strategy_id: str = Field(..., min_length=1)
    configuration_fingerprint: str = Field(default="legacy")
    schema_version: str = Field(default="1.0")
    generation_timestamp: str = Field(default="2026-08-01T00:00:00Z")

    model_config = ConfigDict(frozen=True)


class DecisionResult(BaseModel):
    """Immutable decision outcome containing verdict, confidence, uncertainty, and trace."""

    final_verdict: str = Field(default="ABSTAIN")
    final_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    final_uncertainty: float = Field(default=1.0, ge=0.0, le=1.0)
    escalation_required: bool = Field(default=False)
    explanation_reference: str | None = Field(default=None)
    decision_trace: DecisionTrace = Field(default_factory=DecisionTrace)
    metadata: DecisionMetadata = Field(...)
    action: DecisionAction | str = Field(default=DecisionAction.ABSTAIN)
    rationale: str = Field(default="")
    uncertainty_result: Any = Field(default=None)

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    @model_validator(mode="after")
    def _sync_action_and_verdict(self) -> "DecisionResult":
        if self.final_verdict != "ABSTAIN" and self.action == DecisionAction.ABSTAIN:
            object.__setattr__(self, "action", self.final_verdict)
        elif self.action != DecisionAction.ABSTAIN and self.final_verdict == "ABSTAIN":
            act_str = (
                self.action.value
                if isinstance(self.action, DecisionAction)
                else str(self.action)
            )
            object.__setattr__(self, "final_verdict", act_str)
        return self


class DecisionProfile(BaseModel):
    """Immutable pairing of a profile_id with a decision definition and strategy/engine."""

    profile_id: str = Field(..., min_length=1)
    definition: DecisionDefinition = Field(...)
    strategy: Any = Field(default=None)
    engine: Any = Field(default=None)

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    @model_validator(mode="after")
    def _validate_and_sync(self) -> "DecisionProfile":
        if self.strategy is None and self.engine is not None:
            object.__setattr__(self, "strategy", self.engine)
        elif self.engine is None and self.strategy is not None:
            object.__setattr__(self, "engine", self.strategy)

        target = self.strategy or self.engine
        if hasattr(target, "validate_compatibility"):
            target.validate_compatibility(self.definition)
        return self


class DecisionProfileRegistry(BaseModel):
    """O(1) registry resolver for decision profiles."""

    profiles: tuple[DecisionProfile, ...] = Field(..., min_length=1)

    _profile_index: dict[str, DecisionProfile] = PrivateAttr(default_factory=dict)

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    @model_validator(mode="after")
    def _build_and_validate_index(self) -> "DecisionProfileRegistry":
        from src.core.exceptions import DuplicateDecisionProfileError

        index: dict[str, DecisionProfile] = {}
        for p in self.profiles:
            if p.profile_id in index:
                raise DuplicateDecisionProfileError(
                    f"Duplicate profile_id detected: {p.profile_id}"
                )
            index[p.profile_id] = p
        object.__setattr__(self, "_profile_index", index)
        return self

    def resolve(self, profile_id: str) -> DecisionProfile:
        from src.core.exceptions import DecisionProfileNotFoundError

        if profile_id not in self._profile_index:
            raise DecisionProfileNotFoundError(
                f"Decision profile not found: {profile_id}"
            )
        return self._profile_index[profile_id]


def compute_decision_fingerprint(definition: DecisionDefinition) -> str:
    """Produce a deterministic SHA-256 fingerprint of a DecisionDefinition."""
    canonical = json.dumps(
        {
            "confidence_policy": getattr(definition, "confidence_policy", "calibrated"),
            "decision_strategy": getattr(definition, "decision_strategy", "policy"),
            "escalation_policy": getattr(definition, "escalation_policy", "default"),
            "failure_policy": getattr(definition, "failure_policy", "severity_aware"),
            "uncertainty_policy": getattr(
                definition, "uncertainty_policy", "threshold_based"
            ),
        },
        sort_keys=True,
    )
    return hashlib.sha256(canonical.encode()).hexdigest()
