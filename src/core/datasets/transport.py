"""Immutable transport configuration models for dataset downloads."""

from pydantic import BaseModel, ConfigDict, Field


class FileEndpoint(BaseModel):
    """Configuration for a single file download endpoint."""

    url: str = Field(..., description="Target URL for the file")
    destination_filename: str = Field(..., description="Relative destination filename")
    expected_hash: str = Field(..., description="Expected cryptographic checksum")
    hash_algorithm: str = Field(
        default="sha256", description="Algorithm for verification"
    )

    model_config = ConfigDict(frozen=True)


class AuthenticationMetadata(BaseModel):
    """Configuration for endpoint authentication if required."""

    headers: dict[str, str] = Field(default_factory=dict, description="Auth headers")

    model_config = ConfigDict(frozen=True)


class TransportConfig(BaseModel):
    """Complete transport configuration for a dataset."""

    dataset_id: str = Field(...)
    version: str = Field(...)
    files: tuple[FileEndpoint, ...] = Field(..., min_length=1)
    auth: AuthenticationMetadata | None = Field(default=None)

    model_config = ConfigDict(frozen=True)
