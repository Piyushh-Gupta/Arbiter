"""Immutable domain models and key generation for the Retrieval Caching subsystem."""

import hashlib
import json
import typing
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, model_validator

from src.core.retrieval.retrieval_models import EvidenceBundle

if typing.TYPE_CHECKING:
    from src.core.cache.base import BaseRetrievalCache
else:
    BaseRetrievalCache = typing.Any

__all__ = [
    "CacheDefinition",
    "CacheEntry",
    "CacheKeyGenerator",
    "RetrievalCacheProfile",
    "RetrievalCacheProfileRegistry",
]


class CacheDefinition(BaseModel):
    """Immutable configuration for retrieval result caching."""

    enabled: bool = Field(
        default=True, description="Whether retrieval caching is active."
    )
    backend: str = Field(
        default="in_memory",
        description="Cache backend strategy ('in_memory', 'redis').",
    )
    ttl_seconds: int = Field(
        default=3600, gt=0, description="Time-to-live for cache entries in seconds."
    )
    max_entries: int = Field(
        default=1000, gt=0, description="Maximum entries allowed in cache."
    )
    eviction_policy: str = Field(
        default="lru", description="Eviction policy ('lru', 'fifo')."
    )
    cache_schema_version: str = Field(
        default="1.0", description="Schema version of the cache format."
    )

    model_config = ConfigDict(frozen=True)

    @model_validator(mode="after")
    def _validate_configuration(self) -> "CacheDefinition":
        if self.backend not in ("in_memory", "redis"):
            raise ValueError(f"Unsupported cache backend: '{self.backend}'")
        if self.eviction_policy not in ("lru", "fifo"):
            raise ValueError(f"Unsupported eviction policy: '{self.eviction_policy}'")
        if not self.cache_schema_version:
            raise ValueError("cache_schema_version cannot be empty.")
        return self


class CacheEntry(BaseModel):
    """Immutable structure encapsulating cache metadata and the cached EvidenceBundle."""

    cache_key: str = Field(..., description="Unique SHA-256 cache key.")
    created_at: float = Field(
        ..., description="Epoch timestamp when entry was created."
    )
    expires_at: float | None = Field(
        default=None, description="Epoch timestamp when entry expires."
    )
    corpus_version: str = Field(
        ..., description="Corpus version when entry was generated."
    )
    cache_schema_version: str = Field(..., description="Cache schema version.")
    evidence_bundle: EvidenceBundle = Field(
        ..., description="The cached immutable EvidenceBundle."
    )

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)


class CacheKeyGenerator(BaseModel):
    """
    Immutable, dedicated component responsible for canonical SHA-256 cache-key construction.
    """

    model_config = ConfigDict(frozen=True)

    def generate_key(
        self,
        query: str,
        retrieval_profile_id: str,
        reranking_profile_id: str | None = None,
        retrieval_definition: Any | None = None,
        reranking_definition: Any | None = None,
        corpus_version: str = "default",
        cache_schema_version: str = "1.0",
    ) -> str:
        """
        Generates a deterministic SHA-256 hash string for a retrieval request.
        """
        normalized_query = query.strip().lower()

        def _to_canonical_json(obj: Any) -> str:
            if obj is None:
                return "{}"
            if hasattr(obj, "model_dump_json"):
                return str(obj.model_dump_json())
            if isinstance(obj, dict):
                return json.dumps(obj, sort_keys=True)
            return str(obj)

        canonical_payload = {
            "query": normalized_query,
            "retrieval_profile_id": retrieval_profile_id,
            "reranking_profile_id": reranking_profile_id or "",
            "retrieval_definition": _to_canonical_json(retrieval_definition),
            "reranking_definition": _to_canonical_json(reranking_definition),
            "corpus_version": corpus_version,
            "cache_schema_version": cache_schema_version,
        }
        canonical_str = json.dumps(canonical_payload, sort_keys=True)
        return hashlib.sha256(canonical_str.encode("utf-8")).hexdigest()


class RetrievalCacheProfile(BaseModel):
    """Immutable reusable wrapper binding a cache definition to its backend strategy."""

    profile_id: str = Field(
        ..., description="Unique identifier for this cache profile."
    )
    definition: CacheDefinition = Field(
        ..., description="Immutable configuration for this cache strategy."
    )
    strategy: BaseRetrievalCache = Field(
        ..., description="The executable retrieval cache strategy."
    )

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)


class RetrievalCacheProfileRegistry(BaseModel):
    """Immutable namespace for securely resolving named retrieval cache profiles in O(1) time."""

    profiles: tuple[RetrievalCacheProfile, ...] = Field(
        ...,
        min_length=1,
        description="The collection of registered retrieval cache profiles.",
    )

    _profile_index: dict[str, RetrievalCacheProfile] = PrivateAttr(default_factory=dict)

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    @model_validator(mode="after")
    def _build_and_validate_index(self) -> "RetrievalCacheProfileRegistry":
        from src.core.exceptions import DuplicateCacheProfileError

        index: dict[str, RetrievalCacheProfile] = {}
        for profile in self.profiles:
            if profile.profile_id in index:
                raise DuplicateCacheProfileError(
                    f"Duplicate cache profile identifier: {profile.profile_id}"
                )
            index[profile.profile_id] = profile

        object.__setattr__(self, "_profile_index", index)
        return self

    def resolve(self, profile_id: str) -> RetrievalCacheProfile:
        from src.core.exceptions import CacheProfileNotFoundError

        if profile_id not in self._profile_index:
            raise CacheProfileNotFoundError(
                f"Retrieval cache profile not found: {profile_id}"
            )
        return self._profile_index[profile_id]
