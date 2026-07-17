"""Custom exceptions for the Arbiter project."""


class ArbiterError(Exception):
    """Base exception for all Arbiter domain errors."""


class ConfigurationError(Exception):
    """Raised when there is a configuration or environment error."""

    pass


class RegistryError(Exception):
    """Raised when there is an error interacting with the Dataset Registry."""

    pass


class DownloadError(Exception):
    """Raised when an error occurs during artifact download."""

    pass


class IntegrityError(Exception):
    """Raised when an artifact fails cryptographic integrity verification."""

    pass


class DatasetValidationError(Exception):
    """Raised by the orchestration layer when a dataset fails validation."""

    pass


class VersionNotFoundError(Exception):
    """Raised when a requested dataset version cannot be resolved."""

    pass


class InvalidIdentityError(Exception):
    """Raised when an artifact identity string is malformed or invalid."""

    pass


class ManifestNotFoundError(Exception):
    """Raised when a requested dataset manifest cannot be found."""

    pass


class ManifestParseError(Exception):
    """Raised when a manifest file is malformed or fails schema validation."""

    pass


class UnsupportedSchemaVersionError(Exception):
    """Raised when a manifest uses an unsupported schema version."""

    pass


class ArtifactNotReadyError(Exception):
    """Raised when attempting to load an artifact that is not in the READY lifecycle state."""

    pass


class ArtifactNotFoundError(Exception):
    """Raised when a physical artifact file cannot be found on disk."""

    pass


class UnreadableArtifactError(Exception):
    """Raised when an artifact exists but cannot be read (e.g., permission error, corrupted handle)."""

    pass


class PathTraversalError(Exception):
    """Raised when an artifact filename attempts to escape the isolated version directory."""

    pass


class UnsupportedFormatError(Exception):
    """Raised when the parser registry cannot resolve a requested ParserFormat."""

    pass


class ParserSyntaxError(Exception):
    """Raised when a specific parser strategy encounters syntactically invalid data."""

    pass


class NormalizationFailureError(Exception):
    """Raised when a syntactically parsed record cannot be converted into a normalized canonical record."""

    pass


class MalformedNormalizedRecordError(Exception):
    """Raised when a normalized record fails structural validation upon creation."""

    pass


class UnsupportedTaskSchemaError(Exception):
    """Raised when the mapping registry cannot resolve a requested task schema type."""

    pass


class SchemaMappingError(Exception):
    """Raised when a schema mapping transformation fails unexpectedly."""

    pass


class MissingRequiredFieldError(Exception):
    """Raised when a task record is missing a structurally required field based on the schema mapping."""

    pass


class FieldResolutionError(Exception):
    """Raised when a FieldSelector cannot extract an attribute from a TaskRecord."""

    pass


class FilterConfigurationError(Exception):
    """Raised when a filter step is invalidly configured."""

    pass


class FilterExecutionError(Exception):
    """Raised when the filter pipeline execution encounters an unexpected failure."""

    pass


class PartitionConfigurationError(Exception):
    """Raised when a partition mapping is invalidly configured."""

    pass


class PartitionAssignmentError(Exception):
    """Raised when a task record cannot be assigned to any partition."""

    pass


class PartitionExecutionError(Exception):
    """Raised when the partitioning layer encounters an unexpected failure."""

    pass


class PreprocessingConfigurationError(Exception):
    """Raised when a preprocessing definition and strategy are incompatible."""

    pass


class PreprocessingExecutionError(Exception):
    """Raised when the preprocessing pipeline encounters an unexpected failure."""

    pass


class DuplicatePreprocessingProfileError(Exception):
    """Raised when a PreprocessingProfileRegistry encounters a duplicate profile_id."""

    pass


class PreprocessingProfileNotFoundError(Exception):
    """Raised when a requested PreprocessingProfile cannot be resolved."""

    pass


class ValidationConfigurationError(Exception):
    """Raised when a validation definition and strategy are incompatible."""

    pass


class ValidationProfileNotFoundError(Exception):
    """Raised when a requested validation profile cannot be found."""

    pass


class DuplicateValidationProfileError(Exception):
    """Raised when attempting to register a validation profile with an existing identifier."""

    pass


class ValidationExecutionError(Exception):
    """Raised when the validation pipeline encounters an unexpected failure."""

    pass


class SerializationConfigurationError(Exception):
    """Raised when a serialization definition and strategy are incompatible."""

    pass


class SerializationExecutionError(Exception):
    """Raised when the serialization pipeline encounters an unexpected failure."""

    pass


class DuplicateSerializationProfileError(Exception):
    """Raised when a SerializationProfileRegistry encounters a duplicate profile_id."""

    pass


class SerializationProfileNotFoundError(Exception):
    """Raised when a requested SerializationProfile cannot be resolved."""

    pass


class ExportConfigurationError(Exception):
    """Raised when an export definition and strategy are incompatible."""

    pass


class ExportExecutionError(ArbiterError):
    """Raised when an export pipeline encounters a transport or IO failure."""


class DuplicateExportProfileError(ArbiterError):
    """Raised when an export profile identifier is duplicated within a registry."""


class ExportProfileNotFoundError(ArbiterError):
    """Raised when an export profile identifier cannot be resolved."""

    pass
