"""Custom exceptions for the Arbiter project."""


class ArbiterError(Exception):
    """Base exception for all Arbiter domain errors."""


class ConfigurationError(ArbiterError):
    """Raised when there is a configuration or environment error."""

    pass


class RegistryError(ArbiterError):
    """Raised when there is an error interacting with the Dataset Registry."""

    pass


class DownloadError(ArbiterError):
    """Raised when an error occurs during artifact download."""

    pass


class IntegrityError(ArbiterError):
    """Raised when an artifact fails cryptographic integrity verification."""

    pass


class DatasetValidationError(ArbiterError):
    """Raised by the orchestration layer when a dataset fails validation."""

    pass


class VersionNotFoundError(ArbiterError):
    """Raised when a requested dataset version cannot be resolved."""

    pass


class InvalidIdentityError(ArbiterError):
    """Raised when an artifact identity string is malformed or invalid."""

    pass


class ManifestNotFoundError(ArbiterError):
    """Raised when a requested dataset manifest cannot be found."""

    pass


class ManifestParseError(ArbiterError):
    """Raised when a manifest file is malformed or fails schema validation."""

    pass


class UnsupportedSchemaVersionError(ArbiterError):
    """Raised when a manifest uses an unsupported schema version."""

    pass


class ArtifactNotReadyError(ArbiterError):
    """Raised when attempting to load an artifact that is not in the READY lifecycle state."""

    pass


class ArtifactNotFoundError(ArbiterError):
    """Raised when a physical artifact file cannot be found on disk."""

    pass


class UnreadableArtifactError(ArbiterError):
    """Raised when an artifact exists but cannot be read (e.g., permission error, corrupted handle)."""

    pass


class PathTraversalError(ArbiterError):
    """Raised when an artifact filename attempts to escape the isolated version directory."""

    pass


class UnsupportedFormatError(ArbiterError):
    """Raised when the parser registry cannot resolve a requested ParserFormat."""

    pass


class ParserSyntaxError(ArbiterError):
    """Raised when a specific parser strategy encounters syntactically invalid data."""

    pass


class NormalizationFailureError(ArbiterError):
    """Raised when a syntactically parsed record cannot be converted into a normalized canonical record."""

    pass


class MalformedNormalizedRecordError(ArbiterError):
    """Raised when a normalized record fails structural validation upon creation."""

    pass


class UnsupportedTaskSchemaError(ArbiterError):
    """Raised when the mapping registry cannot resolve a requested task schema type."""

    pass


class SchemaMappingError(ArbiterError):
    """Raised when a schema mapping transformation fails unexpectedly."""

    pass


class MissingRequiredFieldError(ArbiterError):
    """Raised when a task record is missing a structurally required field based on the schema mapping."""

    pass


class FieldResolutionError(ArbiterError):
    """Raised when a FieldSelector cannot extract an attribute from a TaskRecord."""

    pass


class FilterConfigurationError(ArbiterError):
    """Raised when a filter step is invalidly configured."""

    pass


class FilterExecutionError(ArbiterError):
    """Raised when the filter pipeline execution encounters an unexpected failure."""

    pass


class PartitionConfigurationError(ArbiterError):
    """Raised when a partition mapping is invalidly configured."""

    pass


class PartitionAssignmentError(ArbiterError):
    """Raised when a task record cannot be assigned to any partition."""

    pass


class PartitionExecutionError(ArbiterError):
    """Raised when the partitioning layer encounters an unexpected failure."""

    pass


class PreprocessingConfigurationError(ArbiterError):
    """Raised when a preprocessing definition and strategy are incompatible."""

    pass


class PreprocessingExecutionError(ArbiterError):
    """Raised when the preprocessing pipeline encounters an unexpected failure."""

    pass


class DuplicatePreprocessingProfileError(ArbiterError):
    """Raised when a PreprocessingProfileRegistry encounters a duplicate profile_id."""

    pass


class PreprocessingProfileNotFoundError(ArbiterError):
    """Raised when a requested PreprocessingProfile cannot be resolved."""

    pass


class ValidationConfigurationError(ArbiterError):
    """Raised when a validation definition and strategy are incompatible."""

    pass


class ValidationProfileNotFoundError(ArbiterError):
    """Raised when a requested validation profile cannot be found."""

    pass


class DuplicateValidationProfileError(ArbiterError):
    """Raised when attempting to register a validation profile with an existing identifier."""

    pass


class ValidationExecutionError(ArbiterError):
    """Raised when the validation pipeline encounters an unexpected failure."""

    pass


class SerializationConfigurationError(ArbiterError):
    """Raised when a serialization definition and strategy are incompatible."""

    pass


class SerializationExecutionError(ArbiterError):
    """Raised when the serialization pipeline encounters an unexpected failure."""

    pass


class DuplicateSerializationProfileError(ArbiterError):
    """Raised when a SerializationProfileRegistry encounters a duplicate profile_id."""

    pass


class SerializationProfileNotFoundError(ArbiterError):
    """Raised when a requested SerializationProfile cannot be resolved."""

    pass


class ExportConfigurationError(ArbiterError):
    """Raised when an export definition and strategy are incompatible."""

    pass


class ExportExecutionError(ArbiterError):
    """Raised when an export pipeline encounters a transport or IO failure."""


class DuplicateExportProfileError(ArbiterError):
    """Raised when an export profile identifier is duplicated within a registry."""


class ExportProfileNotFoundError(ArbiterError):
    """Raised when an export profile identifier cannot be resolved."""

    pass
