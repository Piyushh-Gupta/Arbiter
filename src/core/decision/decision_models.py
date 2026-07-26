"""Immutable domain models for the Decision Engine subsystem."""

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field

from src.core.uncertainty.uncertainty_models import UncertaintyResult


class DecisionAction(str, Enum):
    """Closed vocabulary of final routing actions."""

    ACCEPT = "ACCEPT"
    REJECT = "REJECT"
    ESCALATE = "ESCALATE"
    ABSTAIN = "ABSTAIN"


class DecisionMetadata(BaseModel):
    """Minimal immutable execution provenance attached to each DecisionResult."""

    strategy_id: str = Field(
        ...,
        description="Identifies which decision engine produced this result.",
    )

    model_config = ConfigDict(frozen=True)


class DecisionDefinition(BaseModel):
    """Base immutable configuration for a decision policy."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)


class DecisionResult(BaseModel):
    """Immutable, self-contained output of a single decision invocation."""

    action: DecisionAction = Field(
        ...,
        description="The deterministic final routing action.",
    )
    rationale: str = Field(
        ...,
        description="A deterministic, human-readable justification for the selected action.",
    )
    uncertainty_result: UncertaintyResult = Field(
        ...,
        description="The unbroken chain of prior pipeline state.",
    )
    metadata: DecisionMetadata = Field(
        ...,
        description="Minimal execution provenance for downstream observability.",
    )

    model_config = ConfigDict(frozen=True)
