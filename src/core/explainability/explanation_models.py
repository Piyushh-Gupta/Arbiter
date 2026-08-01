"""New immutable models for Verification Explainability (M2.7)."""

from pydantic import BaseModel, ConfigDict, Field

from src.core.explainability.explainability_models import ExplanationDefinition


class ContributionAnalysis(BaseModel):
    """Captures numeric contributions of each pipeline stage to the final outcome."""

    retrieval_contribution: float = Field(
        ..., description="Contribution weight of the retrieval stage."
    )
    verification_contribution: float = Field(
        ..., description="Contribution weight of the verification stage."
    )
    aggregation_contribution: float = Field(
        ..., description="Contribution weight of the aggregation stage."
    )
    calibration_contribution: float = Field(
        ..., description="Contribution weight of the calibration stage."
    )

    model_config = ConfigDict(frozen=True)


class ExplanationTrace(BaseModel):
    """Maintains auditable, reproducible verification explanation execution logs."""

    explanation_strategy: str = Field(
        ..., description="Name/type of explanation strategy executed."
    )
    verification_profile: str = Field(
        ..., description="The verification profile ID executed."
    )
    aggregation_profile: str = Field(
        ..., description="The aggregation profile identifier."
    )
    calibration_profile: str = Field(..., description="The calibration profile ID.")
    evidence_traversal: tuple[str, ...] = Field(
        ..., description="Ordered sequence of evidence passage span IDs traversed."
    )
    execution_order: tuple[str, ...] = Field(
        ..., description="Sequence of internal explanation steps executed."
    )

    model_config = ConfigDict(frozen=True)


class EvidenceAttribution(BaseModel):
    """Categorization of evidence passage contributions to the verifier decision."""

    supporting_passages: tuple[str, ...] = Field(
        ..., description="IDs of passages that supported the claim."
    )
    contradicting_passages: tuple[str, ...] = Field(
        ..., description="IDs of passages that refuted/contradicted the claim."
    )
    ignored_passages: tuple[str, ...] = Field(
        ..., description="IDs of passages that did not influence the decision."
    )
    contribution_weights: dict[str, float] = Field(
        ..., description="Numerical weights of passage contributions mapped by span ID."
    )

    model_config = ConfigDict(frozen=True)


class DecisionTrace(BaseModel):
    """Trace records explaining state changes from verification through calibration."""

    aggregation_strategy: str = Field(
        ..., description="Strategy name used for claim aggregation."
    )
    calibration_strategy: str = Field(
        ..., description="Strategy name used for confidence calibration."
    )
    confidence_evolution: tuple[float, ...] = Field(
        ...,
        description="Sequence of confidences (e.g. raw NLI, aggregated, calibrated).",
    )
    contradiction_resolution: str = Field(
        ..., description="Details regarding how conflicting evidence was resolved."
    )

    model_config = ConfigDict(frozen=True)


class VerificationExplanationDefinition(ExplanationDefinition):
    """Configuration definition for M2.7 verification explainers."""

    explanation_strategy: str = Field(
        ..., description="The identifier of explanation strategy."
    )
    verbosity: str = Field(
        default="MEDIUM", description="Detail level (LOW, MEDIUM, HIGH)."
    )
    include_attribution: bool = Field(
        default=True, description="Whether to compute and include evidence attribution."
    )
    include_calibration: bool = Field(
        default=True, description="Whether to include calibration adjustment info."
    )
    include_confidence: bool = Field(
        default=True,
        description="Whether to include confidence progression explanations.",
    )

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)
