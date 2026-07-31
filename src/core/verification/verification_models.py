"""Immutable domain models for the Verification subsystem."""

import math
from datetime import datetime
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


class ExecutionDevice(str, Enum):
    """Supported execution devices for verification runtime."""

    CPU = "CPU"
    CUDA = "CUDA"
    MPS = "MPS"
    TPU = "TPU"
    OTHER = "OTHER"


class ProbabilitySchema(BaseModel):
    """Immutable schema describing expected probability distribution."""

    supported_labels: tuple[str, ...] = Field(
        default=("SUPPORTED", "CONTRADICTED", "INSUFFICIENT"),
        description="Ordered list of supported labels.",
    )
    probability_ordering: tuple[str, ...] = Field(
        default=("SUPPORTED", "CONTRADICTED", "INSUFFICIENT"),
        description="Mapping / ordering of probabilities.",
    )
    tolerance: float = Field(default=1e-5, description="Sum validation tolerance.")

    model_config = ConfigDict(frozen=True)


class NLILabelSchema(BaseModel):
    """Immutable NLI label schema containing mappings and label ordering specifications."""

    label_ordering: tuple[str, ...] = Field(
        default=("CONTRADICTED", "SUPPORTED", "INSUFFICIENT"),
        description="Ordering of labels returned by the NLI model logits.",
    )
    id_mapping: dict[int, str] = Field(
        default_factory=lambda: {0: "CONTRADICTED", 1: "SUPPORTED", 2: "INSUFFICIENT"},
        description="Raw output index to label string mapping.",
    )
    verdict_mapping: dict[str, VerificationVerdict] = Field(
        default_factory=lambda: {
            "SUPPORTED": VerificationVerdict.SUPPORTED,
            "CONTRADICTED": VerificationVerdict.CONTRADICTED,
            "INSUFFICIENT": VerificationVerdict.INSUFFICIENT,
            "CONTRADICTION": VerificationVerdict.CONTRADICTED,
            "ENTAILMENT": VerificationVerdict.SUPPORTED,
            "NEUTRAL": VerificationVerdict.INSUFFICIENT,
            "SUPPORTS": VerificationVerdict.SUPPORTED,
            "REFUTES": VerificationVerdict.CONTRADICTED,
            "NOT_ENOUGH_INFO": VerificationVerdict.INSUFFICIENT,
        },
        description="Mapping from label strings to canonical VerificationVerdict enums.",
    )

    model_config = ConfigDict(frozen=True)

    @model_validator(mode="after")
    def validate_schema(self) -> "NLILabelSchema":
        expected = {
            VerificationVerdict.SUPPORTED,
            VerificationVerdict.CONTRADICTED,
            VerificationVerdict.INSUFFICIENT,
        }
        mapped_verdicts = set(self.verdict_mapping.values())
        if not expected.issubset(mapped_verdicts):
            raise ValueError(
                "verdict_mapping must map to all canonical VerificationVerdict values."
            )
        return self


class NLIModelDefinition(BaseModel):
    """Immutable configuration profile definition for a Natural Language Inference (NLI) model."""

    model_id: str = Field(
        ..., alias="model_identifier", description="HuggingFace model ID."
    )
    tokenizer_id: str = Field(
        ..., alias="tokenizer_identifier", description="HuggingFace tokenizer ID."
    )
    execution_device: ExecutionDevice = Field(
        default=ExecutionDevice.CPU, description="Hardware target."
    )
    inference_precision: str = Field(
        default="fp32", description="Execution precision, e.g. fp16 or fp32."
    )
    max_sequence_length: int = Field(
        default=512, description="Maximum sequence length."
    )
    batch_size: int = Field(default=8, description="Inference batch size.")

    model_config = ConfigDict(frozen=True, populate_by_name=True)

    @property
    def model_identifier(self) -> str:
        return self.model_id

    @property
    def tokenizer_identifier(self) -> str:
        return self.tokenizer_id


class PassageVerificationScore(BaseModel):
    """Immutable verification score container."""

    entailment_probability: float = Field(..., description="Probability of entailment.")
    contradiction_probability: float = Field(
        ..., description="Probability of contradiction."
    )
    neutral_probability: float = Field(..., description="Probability of neutral.")

    model_config = ConfigDict(frozen=True)

    @model_validator(mode="before")
    @classmethod
    def validate_probabilities_before(cls, data: Any) -> Any:
        if isinstance(data, dict):
            for k, v in data.items():
                if isinstance(v, (int, float)):
                    fv = float(v)
                    if math.isnan(fv):
                        raise ValueError(f"Value for {k} cannot be NaN.")
                    if math.isinf(fv):
                        raise ValueError(f"Value for {k} cannot be infinite.")
        return data

    @model_validator(mode="after")
    def validate_probabilities(self) -> "PassageVerificationScore":
        for name, val in [
            ("entailment_probability", self.entailment_probability),
            ("contradiction_probability", self.contradiction_probability),
            ("neutral_probability", self.neutral_probability),
        ]:
            if math.isnan(val) or math.isinf(val):
                raise ValueError(f"Value for {name} must be finite.")
            if not (0.0 <= val <= 1.0):
                raise ValueError(f"Value for {name} must be in [0.0, 1.0], got {val}")

        total = (
            self.entailment_probability
            + self.contradiction_probability
            + self.neutral_probability
        )
        if abs(total - 1.0) > 1e-5:
            raise ValueError(
                f"Probabilities must sum to 1.0 (with 1e-5 tolerance), got {total}"
            )
        return self

    def conforms_to_schema(self, schema: ProbabilitySchema) -> bool:
        """Checks if this score conforms to the active schema."""
        total = (
            self.entailment_probability
            + self.contradiction_probability
            + self.neutral_probability
        )
        if abs(total - 1.0) > schema.tolerance:
            return False
        for val in [
            self.entailment_probability,
            self.contradiction_probability,
            self.neutral_probability,
        ]:
            if not (0.0 <= val <= 1.0) or math.isnan(val) or math.isinf(val):
                return False
        return True

    def validate_against_schema(self, schema: ProbabilitySchema) -> None:
        """Validates that this score conforms to the given ProbabilitySchema."""
        if not self.conforms_to_schema(schema):
            raise ValueError(
                "PassageVerificationScore does not conform to the active ProbabilitySchema."
            )


class PassageVerificationMetadata(BaseModel):
    """Strongly typed metadata for passage verification."""

    model_version: str = Field(default="1.0", description="Model version.")
    inference_precision: str = Field(default="fp32", description="Inference precision.")
    device_used: ExecutionDevice = Field(
        default=ExecutionDevice.CPU, description="Device used."
    )

    model_config = ConfigDict(frozen=True)


class PassageVerificationResult(BaseModel):
    """Immutable per-passage verification outcome."""

    span_id: str = Field(
        ..., description="Unique span identifier of the evidence passage."
    )
    verdict: VerificationVerdict = Field(..., description="Passage-level verdict.")
    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Passage-level confidence score."
    )
    probability_distribution: PassageVerificationScore = Field(
        ..., description="Verification probability distribution."
    )
    rationale: str | None = Field(
        default=None, description="Optional rationale for passage verdict."
    )
    latency: float | None = Field(default=None, description="Latency in seconds.")
    metadata: PassageVerificationMetadata = Field(
        default_factory=PassageVerificationMetadata,
        description="Strongly typed passage verification metadata.",
    )

    model_config = ConfigDict(frozen=True)

    @property
    def raw_scores(self) -> dict[str, float]:
        """Backward compatibility helper property."""
        return {
            "SUPPORTED": self.probability_distribution.entailment_probability,
            "CONTRADICTED": self.probability_distribution.contradiction_probability,
            "INSUFFICIENT": self.probability_distribution.neutral_probability,
        }


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
    probability_schema: ProbabilitySchema = Field(
        default_factory=ProbabilitySchema,
        description="Active probability schema description.",
    )

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)


class NLIVerificationDefinition(VerificationDefinition):
    """Immutable configuration specifically for NLI verification execution."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)


class VerificationExecutionMetadata(BaseModel):
    """Immutable metadata model describing the execution specifics of a verification request."""

    request_id: str = Field(..., description="Execution request identifier.")
    execution_duration: float = Field(..., description="Execution duration in seconds.")
    verifier_profile: str = Field(..., description="Name of the verifier profile.")
    aggregation_profile: str = Field(
        ..., description="Name of the aggregation profile."
    )
    configuration_fingerprint: str = Field(
        ..., description="SHA-256 fingerprint of the configuration."
    )

    model_config = ConfigDict(frozen=True)


class PassageVerificationInput(BaseModel):
    """Immutable input arguments required to verify a single evidence passage."""

    claim: str = Field(..., description="The textual assertion to evaluate.")
    passage: Any = Field(..., description="The evidence passage object.")
    execution_metadata: VerificationExecutionMetadata = Field(
        ..., description="Execution provenance."
    )

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)


class ClaimVerificationInput(BaseModel):
    """Immutable input arguments required to verify a claim against a bundle of evidence."""

    claim: str = Field(..., description="The textual assertion to evaluate.")
    bundle: Any = Field(..., description="The evidence bundle containing passages.")
    definition: VerificationDefinition = Field(
        ..., description="Execution configuration options."
    )

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)


class AggregationMetadata(BaseModel):
    """Immutable metadata summarizing the aggregation details."""

    strategy_id: str = Field(..., description="The aggregation strategy used.")
    thresholds_applied: dict[str, float] = Field(
        default_factory=dict, description="Configuration thresholds applied."
    )

    model_config = ConfigDict(frozen=True)


class ClaimVerificationContext(BaseModel):
    """Immutable context encapsulating the intermediate elements of verification execution."""

    ordered_passage_results: tuple[PassageVerificationResult, ...] = Field(
        ..., description="Ordered passage outcomes."
    )
    aggregation_metadata: AggregationMetadata = Field(
        ..., description="Strongly typed aggregation metadata."
    )
    execution_metadata: VerificationExecutionMetadata = Field(
        ..., description="Execution provenance."
    )

    model_config = ConfigDict(frozen=True)


class VerifierRuntimeMetadata(BaseModel):
    """Immutable environment and model identifier information."""

    model_id: str = Field(..., description="Unique model identifier.")
    revision: str = Field(..., description="Git commit hash or revision.")
    tokenizer: str = Field(..., description="Tokenizer identifier.")
    framework: str = Field(..., description="Deep learning framework used.")
    execution_device: ExecutionDevice = Field(
        ..., description="Platform hardware target."
    )
    inference_precision: str = Field(
        ..., description="Precision string, e.g. fp16, fp32."
    )
    execution_timestamp: datetime = Field(
        ..., description="Timezone-aware execution timestamp."
    )

    model_config = ConfigDict(frozen=True)

    @model_validator(mode="after")
    def validate_timezone_aware(self) -> "VerifierRuntimeMetadata":
        if (
            self.execution_timestamp.tzinfo is None
            or self.execution_timestamp.tzinfo.utcoffset(self.execution_timestamp)
            is None
        ):
            raise ValueError("execution_timestamp must be timezone-aware.")
        return self

    @property
    def model_identifier(self) -> str:
        """Backward compatibility helper property."""
        return self.model_id


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
    metadata_provider: Any = Field(
        default=None,
        description="The metadata provider strategy associated with this profile.",
    )

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    @model_validator(mode="after")
    def _validate_compatibility(self) -> "VerificationProfile":
        # 1. Verifier compatibility
        self.verifier.validate_compatibility(self.definition)

        # 2. Aggregation strategy compatibility
        from src.core.verification.aggregation import BaseAggregationStrategy

        agg = self.definition.aggregation_strategy
        if agg is not None and not isinstance(agg, str):
            if not isinstance(agg, BaseAggregationStrategy):
                raise ValueError(
                    "aggregation_strategy must implement BaseAggregationStrategy protocol."
                )

        # 3. Probability schema compatibility
        if self.definition.probability_schema is None:
            raise ValueError("probability_schema in definition cannot be None.")

        # 4. Metadata provider compatibility
        from src.core.verification.base import BaseMetadataProvider

        if self.metadata_provider is not None:
            if not isinstance(self.metadata_provider, BaseMetadataProvider):
                raise ValueError(
                    "metadata_provider must implement BaseMetadataProvider protocol."
                )

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
