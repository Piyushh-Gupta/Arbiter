"""Immutable configuration models for the serialization framework."""

import typing

from pydantic import BaseModel, ConfigDict, Field, model_validator

if typing.TYPE_CHECKING:
    from src.core.datasets.serialization.base import BaseSerializer
else:
    BaseSerializer = typing.Any


class SerializationDefinition(BaseModel):
    """Base immutable configuration for a serialization target."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)


class SerializationStep(BaseModel):
    """Binds a SerializationDefinition to its executable BaseSerializer strategy."""

    definition: SerializationDefinition = Field(
        ...,
        description="The strictly immutable configuration for this serialization step.",
    )
    strategy: BaseSerializer = Field(
        ...,
        description="The stateless executable strategy resolving the definition.",
    )

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    @model_validator(mode="after")
    def _validate_compatibility(self) -> "SerializationStep":
        """Statically verifies compatibility between the definition and strategy."""
        self.strategy.validate_compatibility(self.definition)
        return self


class SerializationPipeline(BaseModel):
    """An ordered sequence of SerializationSteps to execute."""

    steps: tuple[SerializationStep, ...] = Field(
        ...,
        description="The ordered collection of strictly bound serialization steps.",
    )

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)
