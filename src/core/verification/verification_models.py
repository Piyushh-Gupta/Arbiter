"""Immutable domain models for the Verification subsystem."""

from enum import Enum
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, model_validator

if TYPE_CHECKING:
    from src.core.retrieval.retrieval_models import EvidenceBundle, EvidencePassage
    from src.core.verification.base import BaseVerifier
else:
    EvidenceBundle = Any
    EvidencePassage = Any
    BaseVerifier = Any


class VerificationVerdict(str, Enum):
    """Strongly-typed closed vocabulary of verification verdicts."""

    SUPPORTED = "SUPPORTED"
    CONTRADICTED = "CONTRADICTED"
    INSUFFICIENT = "INSUFFICIENT"


class VerificationLabel(str, Enum):
    """Closed vocabulary of verification outcomes for legacy M1 consumers."""

    SUPPORTS = "SUPPORTS"
    REFUTES = "REFUTES"
    NOT_ENOUGH_INFO = "NOT_ENOUGH_INFO"


VERDICT_TO_LABEL: dict[VerificationVerdict, VerificationLabel] = {
    VerificationVerdict.SUPPORTED: VerificationLabel.SUPPORTS,
    VerificationVerdict.CONTRADICTED: VerificationLabel.REFUTES,
    VerificationVerdict.INSUFFICIENT: VerificationLabel.NOT_ENOUGH_INFO,
}

LABEL_TO_VERDICT: dict[Any, VerificationVerdict] = {
    VerificationLabel.SUPPORTS: VerificationVerdict.SUPPORTED,
    VerificationLabel.REFUTES: VerificationVerdict.CONTRADICTED,
    VerificationLabel.NOT_ENOUGH_INFO: VerificationVerdict.INSUFFICIENT,
    "SUPPORTS": VerificationVerdict.SUPPORTED,
    "REFUTES": VerificationVerdict.CONTRADICTED,
    "NOT_ENOUGH_INFO": VerificationVerdict.INSUFFICIENT,
    "SUPPORTED": VerificationVerdict.SUPPORTED,
    "CONTRADICTED": VerificationVerdict.CONTRADICTED,
    "INSUFFICIENT": VerificationVerdict.INSUFFICIENT,
    VerificationVerdict.SUPPORTED: VerificationVerdict.SUPPORTED,
    VerificationVerdict.CONTRADICTED: VerificationVerdict.CONTRADICTED,
    VerificationVerdict.INSUFFICIENT: VerificationVerdict.INSUFFICIENT,
}


class PassageVerificationResult(BaseModel):
    """Immutable per-passage verification outcome."""

    span_id: str = Field(
        ..., description="Unique span identifier of the evidence passage."
    )
    verdict: VerificationVerdict = Field(..., description="Passage-level verdict.")
    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Passage-level confidence score."
    )
    raw_scores: dict[str, float] | None = Field(
        default=None, description="Optional raw class scores."
    )
    rationale: str | None = Field(
        default=None, description="Optional rationale for passage verdict."
    )

    model_config = ConfigDict(frozen=True)


class VerifiedPassage(BaseModel):
    """Legacy immutable binding of an EvidencePassage to per-passage verification scores."""

    passage: Any = Field(..., description="Source evidence passage.")
    label: Any = Field(..., description="Winning discrete verification label.")
    supports_score: float = Field(..., ge=0.0, le=1.0)
    refutes_score: float = Field(..., ge=0.0, le=1.0)
    not_enough_info_score: float = Field(..., ge=0.0, le=1.0)

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)


class VerificationExplanation(BaseModel):
    """Immutable structured explanation of verification reasoning and attribution."""

    reasoning: str = Field(
        default="", description="Human-readable structured reasoning."
    )
    supporting_evidence_references: tuple[str, ...] = Field(
        default_factory=tuple, description="References to supporting span_ids."
    )
    confidence_explanation: str = Field(
        default="", description="Explanation of confidence calculation."
    )
    uncertainty_explanation: str = Field(
        default="", description="Explanation of uncertainty factors."
    )

    model_config = ConfigDict(frozen=True)


class VerificationModelMetadata(BaseModel):
    """Immutable metadata describing the verification model/framework environment."""

    model_identifier: str = Field(
        ..., description="Identifier or path of the verification model."
    )
    revision: str = Field(default="1.0", description="Model revision or commit hash.")
    tokenizer: str = Field(default="default", description="Tokenizer identifier.")
    execution_device: str = Field(
        default="cpu", description="Execution device (e.g. cpu, cuda)."
    )
    framework_version: str = Field(
        default="1.0.0", description="Framework version string."
    )

    model_config = ConfigDict(frozen=True)


class VerificationMetadata(BaseModel):
    """Legacy execution provenance attached to VerificationResult."""

    strategy_id: str = Field(
        ..., description="Identifies which verifier produced this result."
    )

    model_config = ConfigDict(frozen=True)


class VerificationDefinition(BaseModel):
    """Immutable configuration for verification execution."""

    verifier_model: str = Field(
        default="nli-default", description="Verifier model/engine identifier."
    )
    verdict_schema: tuple[VerificationVerdict, ...] = Field(
        default=(
            VerificationVerdict.SUPPORTED,
            VerificationVerdict.CONTRADICTED,
            VerificationVerdict.INSUFFICIENT,
        ),
        description="Allowed verdict schema.",
    )
    aggregation_strategy: Any = Field(
        default="max_confidence",
        description="Aggregation strategy instance or identifier.",
    )
    confidence_thresholds: dict[str, float] = Field(
        default_factory=lambda: {"SUPPORTED": 0.5, "CONTRADICTED": 0.5},
        description="Confidence thresholds required for verdicts.",
    )
    reasoning_enabled: bool = Field(
        default=True, description="Enable reasoning generation."
    )
    explanation_enabled: bool = Field(
        default=True, description="Enable structured explainability."
    )
    top_k: int = Field(
        default=5, gt=0, description="Maximum passages to evaluate from bundle."
    )

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)


class NLIVerificationDefinition(VerificationDefinition):
    """Immutable configuration specifically for NLI verification execution."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)


class VerificationResult(BaseModel):
    """Immutable, self-contained outcome of claim verification."""

    verdict: VerificationVerdict = Field(
        default=VerificationVerdict.INSUFFICIENT,
        description="The final discrete verdict.",
    )
    confidence: float | None = Field(
        default=0.0, ge=0.0, le=1.0, description="Overall confidence score."
    )
    supporting_passages: tuple[str, ...] = Field(
        default_factory=tuple, description="span_ids of supporting passages."
    )
    contradicting_passages: tuple[str, ...] = Field(
        default_factory=tuple, description="span_ids of contradicting passages."
    )
    evidence_attribution: dict[str, float] = Field(
        default_factory=dict, description="Mapping of span_id to attribution score."
    )
    explanation: VerificationExplanation = Field(
        default_factory=VerificationExplanation,
        description="Structured verification explanation.",
    )
    uncertainty: float | None = Field(
        default=None, ge=0.0, le=1.0, description="Estimated uncertainty score."
    )
    model_metadata: VerificationModelMetadata = Field(
        default_factory=lambda: VerificationModelMetadata(
            model_identifier="nli-default"
        ),
        description="Execution model environment metadata.",
    )

    # Legacy fields for backward compatibility with M1 pipeline
    label: Any = Field(default=None, description="Legacy verdict label.")
    evidence_bundle: Any | None = Field(default=None)
    verified_passages: tuple[Any, ...] | None = Field(default=None)
    metadata: VerificationMetadata | None = Field(default=None)

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    @model_validator(mode="before")
    @classmethod
    def _preprocess_legacy_fields(cls, data: Any) -> Any:
        if isinstance(data, dict):
            if "label" in data and ("verdict" not in data or data["verdict"] is None):
                lbl = data["label"]
                if lbl in LABEL_TO_VERDICT:
                    data["verdict"] = LABEL_TO_VERDICT[lbl]
                elif isinstance(lbl, VerificationVerdict):
                    data["verdict"] = lbl
        return data

    @model_validator(mode="after")
    def _sync_legacy_label(self) -> "VerificationResult":
        expected_lbl = VERDICT_TO_LABEL.get(
            self.verdict, VerificationLabel.NOT_ENOUGH_INFO
        )
        if self.label != expected_lbl:
            object.__setattr__(self, "label", expected_lbl)
        return self


class VerificationProfile(BaseModel):
    """Immutable wrapper binding a verification definition to its execution strategy."""

    profile_id: str = Field(
        ..., description="Unique identifier for this verification profile."
    )
    definition: VerificationDefinition = Field(
        ..., description="Immutable configuration definition."
    )
    verifier: BaseVerifier = Field(
        ..., description="The executable verifier strategy resolving the definition."
    )

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    @model_validator(mode="after")
    def _validate_compatibility(self) -> "VerificationProfile":
        self.verifier.validate_compatibility(self.definition)
        return self


class VerificationProfileRegistry(BaseModel):
    """Immutable O(1) registry for named verification profiles."""

    profiles: tuple[VerificationProfile, ...] = Field(
        ..., min_length=1, description="Collection of registered verification profiles."
    )

    _profile_index: dict[str, VerificationProfile] = PrivateAttr(default_factory=dict)

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    @model_validator(mode="after")
    def _build_and_validate_index(self) -> "VerificationProfileRegistry":
        from src.core.exceptions import DuplicateVerificationProfileError

        index: dict[str, VerificationProfile] = {}
        for profile in self.profiles:
            if profile.profile_id in index:
                raise DuplicateVerificationProfileError(
                    f"Duplicate verification profile identifier: {profile.profile_id}"
                )
            index[profile.profile_id] = profile

        object.__setattr__(self, "_profile_index", index)
        return self

    def resolve(self, profile_id: str) -> VerificationProfile:
        from src.core.exceptions import VerificationProfileNotFoundError

        if profile_id not in self._profile_index:
            raise VerificationProfileNotFoundError(
                f"Verification profile not found: {profile_id}"
            )
        return self._profile_index[profile_id]
