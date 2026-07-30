"""Immutable domain models for the Reranking subsystem."""

import typing

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, model_validator

from src.core.retrieval.retrieval_models import RerankMetadata

if typing.TYPE_CHECKING:
    from src.core.reranking.base import BaseReranker
else:
    BaseReranker = typing.Any

__all__ = [
    "CrossEncoderModelMetadata",
    "RerankMetadata",
    "RerankingDefinition",
    "RerankingProfile",
    "RerankingProfileRegistry",
]


class CrossEncoderModelMetadata(BaseModel):
    """Immutable metadata describing a Cross-Encoder model checkpoint and execution parameters."""

    model_identifier: str = Field(
        ..., description="Identifier of the model checkpoint."
    )
    model_revision: str | None = Field(
        default=None, description="Optional revision/version of the model checkpoint."
    )
    tokenizer_identifier: str = Field(
        ..., description="Identifier of the tokenizer used for text encoding."
    )
    inference_framework: str = Field(
        ...,
        description="Framework used for inference (e.g. 'sentence-transformers', 'torch').",
    )
    execution_device: str = Field(
        ..., description="Target execution device (e.g. 'cpu', 'cuda')."
    )
    max_sequence_length: int = Field(
        ..., gt=0, description="Maximum token sequence length supported by the model."
    )

    model_config = ConfigDict(frozen=True)


class RerankingDefinition(BaseModel):
    """Immutable configuration for a single reranking invocation."""

    model_identifier: str = Field(
        default="cross-encoder/ms-marco-MiniLM-L-6-v2",
        description="Identifier of the cross-encoder model checkpoint.",
    )
    batch_size: int = Field(
        default=32,
        gt=0,
        description="Maximum batch size for model inference.",
    )
    top_k_input: int = Field(
        default=10,
        gt=0,
        description="Maximum number of stage-1 candidates accepted for reranking.",
    )
    top_k_output: int = Field(
        default=5,
        gt=0,
        description="Maximum number of reranked candidates to return.",
    )
    top_k: int | None = Field(
        default=None,
        gt=0,
        description="Optional alias for top_k_output for backward compatibility.",
    )
    device: str = Field(
        default="cpu",
        description="Target execution device (e.g. 'cpu', 'cuda').",
    )
    score_threshold: float | None = Field(
        default=None,
        description="Optional minimum cross-encoder score filter.",
    )

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    @model_validator(mode="after")
    def _validate_bounds(self) -> "RerankingDefinition":
        out_k = self.top_k if self.top_k is not None else self.top_k_output
        object.__setattr__(self, "top_k_output", out_k)
        object.__setattr__(self, "top_k", out_k)
        if self.top_k_output > self.top_k_input:
            raise ValueError("top_k_output cannot exceed top_k_input.")
        return self


class RerankingProfile(BaseModel):
    """Immutable reusable wrapper binding a reranking definition to its execution strategy."""

    profile_id: str = Field(
        ..., description="Unique identifier for this reranking profile."
    )
    definition: RerankingDefinition = Field(
        ..., description="Immutable configuration for this reranking strategy."
    )
    strategy: BaseReranker = Field(
        ..., description="The stateless executable strategy resolving the definition."
    )

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    @model_validator(mode="after")
    def _validate_compatibility(self) -> "RerankingProfile":
        self.strategy.validate_compatibility(self.definition)
        return self


class RerankingProfileRegistry(BaseModel):
    """Immutable namespace for securely resolving named reranking profiles in O(1) time."""

    profiles: tuple[RerankingProfile, ...] = Field(
        ...,
        min_length=1,
        description="The collection of registered reranking profiles.",
    )

    _profile_index: dict[str, RerankingProfile] = PrivateAttr(default_factory=dict)

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    @model_validator(mode="after")
    def _build_and_validate_index(self) -> "RerankingProfileRegistry":
        from src.core.exceptions import DuplicateRerankingProfileError

        index: dict[str, RerankingProfile] = {}
        for profile in self.profiles:
            if profile.profile_id in index:
                raise DuplicateRerankingProfileError(
                    f"Duplicate reranking profile identifier: {profile.profile_id}"
                )
            index[profile.profile_id] = profile

        object.__setattr__(self, "_profile_index", index)
        return self

    def resolve(self, profile_id: str) -> RerankingProfile:
        from src.core.exceptions import RerankingProfileNotFoundError

        if profile_id not in self._profile_index:
            raise RerankingProfileNotFoundError(
                f"Reranking profile not found: {profile_id}"
            )
        return self._profile_index[profile_id]
