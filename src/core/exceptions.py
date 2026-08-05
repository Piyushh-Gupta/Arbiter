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


class LoadingConfigurationError(ArbiterError):
    """Raised when a loading strategy receives an incompatible or malformed configuration."""


class LoadingExecutionError(ArbiterError):
    """Raised when dataset reconstruction fails at runtime."""


class DuplicateLoadingProfileError(ArbiterError):
    """Raised when duplicate loading profile identifiers exist during registry construction."""


class LoadingProfileNotFoundError(ArbiterError):
    """Raised when a loading profile cannot be resolved from the registry."""


class RetrievalConfigurationError(ArbiterError):
    """Raised when a retrieval strategy receives an incompatible or malformed definition."""


class RetrievalExecutionError(ArbiterError):
    """Raised when a retrieval strategy encounters a runtime failure."""


class DuplicateRetrievalProfileError(ArbiterError):
    """Raised when a RetrievalProfileRegistry encounters a duplicate profile_id."""


class RetrievalProfileNotFoundError(ArbiterError):
    """Raised when a retrieval profile cannot be resolved from the registry."""


class RerankingConfigurationError(ArbiterError):
    """Raised when a reranking strategy receives an incompatible or malformed definition."""


class RerankingExecutionError(ArbiterError):
    """Raised when a reranking strategy encounters a runtime failure."""


class DuplicateRerankingProfileError(ArbiterError):
    """Raised when a RerankingProfileRegistry encounters a duplicate profile_id."""


class RerankingProfileNotFoundError(ArbiterError):
    """Raised when a reranking profile cannot be resolved from the registry."""


class CacheConfigurationError(ArbiterError):
    """Raised when a cache definition or strategy receives an incompatible configuration."""


class CacheExecutionError(ArbiterError):
    """Raised when a cache operation encounters a runtime failure."""


class DuplicateCacheProfileError(ArbiterError):
    """Raised when a RetrievalCacheProfileRegistry encounters a duplicate profile_id."""


class CacheProfileNotFoundError(ArbiterError):
    """Raised when a cache profile cannot be resolved from the registry."""


class BenchmarkConfigurationError(ArbiterError):
    """Raised when a benchmark definition or strategy receives an incompatible configuration."""


class BenchmarkExecutionError(ArbiterError):
    """Raised when a benchmark evaluation encounters a runtime failure."""


class DuplicateBenchmarkProfileError(ArbiterError):
    """Raised when a BenchmarkProfileRegistry encounters a duplicate profile_id."""


class BenchmarkProfileNotFoundError(ArbiterError):
    """Raised when a benchmark profile cannot be resolved from the registry."""


class DuplicateMetricError(ArbiterError):
    """Raised when a MetricRegistry encounters a duplicate metric name."""


class MetricNotFoundError(ArbiterError):
    """Raised when a metric cannot be resolved from the MetricRegistry."""


class OptimizationConfigurationError(ArbiterError):
    """Raised when an optimization policy or profile receives an invalid configuration."""


class OptimizationExecutionError(ArbiterError):
    """Raised when an optimization operation encounters a runtime failure."""


class OptimizationTimeoutError(ArbiterError):
    """Raised when an optimized retrieval operation exceeds the timeout policy."""


class DuplicateOptimizationProfileError(ArbiterError):
    """Raised when an OptimizationProfileRegistry encounters a duplicate profile_id."""


class OptimizationProfileNotFoundError(ArbiterError):
    """Raised when an optimization profile cannot be resolved from the registry."""


class VerificationConfigurationError(ArbiterError):
    """Raised when a verifier receives an incompatible or malformed definition."""


class VerificationExecutionError(ArbiterError):
    """Raised when a verifier encounters a runtime failure."""


class DuplicateVerificationProfileError(ArbiterError):
    """Raised when a VerificationProfileRegistry encounters a duplicate profile_id."""


class VerificationProfileNotFoundError(ArbiterError):
    """Raised when a verification profile cannot be resolved from the registry."""


class CalibrationConfigurationError(ArbiterError):
    """Raised when a calibrator receives an incompatible or malformed definition."""


class CalibrationExecutionError(ArbiterError):
    """Raised when a calibrator encounters a runtime failure."""


class FailureAnalysisConfigurationError(ArbiterError):
    """Raised when a failure analyzer receives an incompatible or malformed definition."""


class FailureAnalysisExecutionError(ArbiterError):
    """Raised when a failure analyzer encounters a runtime failure."""


class DuplicateFailureAnalysisProfileError(ArbiterError):
    """Raised when a FailureAnalysisProfileRegistry encounters a duplicate profile_id."""


class FailureAnalysisProfileNotFoundError(ArbiterError):
    """Raised when a failure analysis profile cannot be resolved from the registry."""


class UncertaintyConfigurationError(ArbiterError):
    """Raised when an uncertainty estimator receives an incompatible definition."""


class UncertaintyExecutionError(ArbiterError):
    """Raised when an uncertainty estimator encounters a runtime failure."""


class DuplicateUncertaintyProfileError(ArbiterError):
    """Raised when an UncertaintyProfileRegistry encounters a duplicate profile_id."""


class UncertaintyProfileNotFoundError(ArbiterError):
    """Raised when an uncertainty profile cannot be resolved from the registry."""


class DecisionConfigurationError(ArbiterError):
    """Raised when a decision engine configuration is invalid."""


class DecisionExecutionError(ArbiterError):
    """Raised when a decision engine encounters a runtime failure."""


class DuplicateDecisionProfileError(ArbiterError):
    """Raised when a DecisionProfileRegistry encounters a duplicate profile_id."""


class DecisionProfileNotFoundError(ArbiterError):
    """Raised when a decision profile cannot be resolved from the registry."""


class DuplicateDecisionMetricPolicyError(ArbiterError):
    """Raised when duplicate policy IDs are registered in the metric registry."""


class DecisionMetricPolicyNotFoundError(ArbiterError):
    """Raised when a metric policy cannot be resolved from the registry."""


class DuplicateDecisionRiskPolicyError(ArbiterError):
    """Raised when duplicate policy IDs are registered in the risk registry."""


class DecisionRiskPolicyNotFoundError(ArbiterError):
    """Raised when a risk policy cannot be resolved from the registry."""


class DuplicateDecisionBenchmarkProfileError(ArbiterError):
    """Raised when duplicate benchmark profile IDs are registered."""


class DecisionBenchmarkProfileNotFoundError(ArbiterError):
    """Raised when a benchmark profile cannot be resolved from the registry."""


class DuplicateDecisionExplanationProfileError(ArbiterError):
    """Raised when duplicate explanation profile IDs are registered."""


class DecisionExplanationProfileNotFoundError(ArbiterError):
    """Raised when an explanation profile cannot be resolved from the registry."""


class DecisionCacheConfigurationError(ArbiterError):
    """Raised when caching parameters are invalid."""


class DecisionExecutionTimeoutError(ArbiterError):
    """Raised when decision execution times out."""


class DuplicateDecisionOptimizationProfileError(ArbiterError):
    """Raised when duplicate optimization profile IDs are registered."""


class DecisionOptimizationProfileNotFoundError(ArbiterError):
    """Raised when an optimization profile cannot be resolved from the registry."""


class ExplanationConfigurationError(ArbiterError):
    """Raised when an explainer configuration is invalid."""


class ExplanationExecutionError(ArbiterError):
    """Raised when an explainer encounters a runtime failure."""


class DuplicateExplanationProfileError(ArbiterError):
    """Raised when an ExplanationProfileRegistry encounters a duplicate profile_id."""


class ExplanationProfileNotFoundError(ArbiterError):
    """Raised when an explanation profile cannot be resolved from the registry."""


class EvaluationConfigurationError(ArbiterError):
    """Raised when an evaluator configuration is invalid."""


class EvaluationExecutionError(ArbiterError):
    """Raised when an evaluator encounters a runtime failure."""


class DuplicateEvaluationProfileError(ArbiterError):
    """Raised when an EvaluationProfileRegistry encounters a duplicate profile_id."""


class EvaluationProfileNotFoundError(ArbiterError):
    """Raised when an evaluation profile cannot be resolved from the registry."""


class PipelineConfigurationError(ArbiterError):
    """Raised when a PipelineDefinition or PipelineProfile contains invalid configuration."""


class PipelineStageExecutionError(ArbiterError):
    """Raised when a pipeline stage encounters a non-recoverable execution failure."""


class DuplicatePipelineProfileError(ArbiterError):
    """Raised when duplicate profile_id values are registered."""


class PipelineProfileNotFoundError(ArbiterError):
    """Raised when a PipelineProfile cannot be resolved from the registry."""


class TelemetryConfigurationError(ArbiterError):
    """Raised when a telemetry exporter or collector configuration is invalid."""


class TelemetryExecutionError(ArbiterError):
    """Raised when a telemetry exporter encounters a runtime failure."""


class DuplicateTelemetryProfileError(ArbiterError):
    """Raised when a TelemetryExporterRegistry encounters a duplicate profile_id."""


class TelemetryProfileNotFoundError(ArbiterError):
    """Raised when a telemetry exporter profile cannot be resolved from the registry."""


class PipelineResilienceConfigurationError(ArbiterError):
    """Raised when a PipelineResilienceProfile or registry contains invalid configuration."""


class PipelineResilienceTimeoutError(PipelineStageExecutionError):
    """Raised when a pipeline execution exceeds the configured timeout threshold."""


class DuplicateResilienceProfileError(ArbiterError):
    """Raised when a PipelineResilienceProfileRegistry detects a duplicate profile_id."""


class ResilienceProfileNotFoundError(ArbiterError):
    """Raised when a PipelineResilienceProfile cannot be resolved from the registry."""


class PipelineBenchmarkConfigurationError(ArbiterError):
    """Raised when a PipelineBenchmarkProfile or registry contains invalid configuration."""


class DuplicatePipelineBenchmarkProfileError(ArbiterError):
    """Raised when duplicate benchmark profile IDs are registered."""


class PipelineBenchmarkProfileNotFoundError(ArbiterError):
    """Raised when a benchmark profile cannot be resolved from the registry."""


class PipelineExplanationConfigurationError(ArbiterError):
    """Raised when a PipelineExplanationProfile or registry contains invalid configuration."""


class DuplicatePipelineExplanationProfileError(ArbiterError):
    """Raised when a PipelineExplanationProfileRegistry detects a duplicate profile_id."""


class PipelineExplanationProfileNotFoundError(ArbiterError):
    """Raised when a PipelineExplanationProfile cannot be resolved from the registry."""


class PipelineOperationalConfigurationError(PipelineConfigurationError):
    """Raised when a PipelineOperationalProfile contains invalid configuration."""


class DuplicateOperationalProfileError(ArbiterError):
    """Raised when a PipelineOperationalProfileRegistry detects a duplicate profile_id."""


class OperationalProfileNotFoundError(ArbiterError):
    """Raised when a PipelineOperationalProfile cannot be resolved from the registry."""


class IllegalLifecycleTransitionError(ArbiterError):
    """Raised when an illegal transition is attempted in the PipelineLifecycleManager."""


class PipelineOperationalExecutionError(ArbiterError):
    """Raised when an operational controller encounters an execution failure."""


class APIServiceConfigurationError(ConfigurationError):
    """Raised when the API service layer is misconfigured."""


class ApiContractConfigurationError(ConfigurationError):
    """Raised when the API contract subsystem is misconfigured."""


class ApiContractValidationError(ArbiterError):
    """Raised when an API request or response fails structural validation."""


class DuplicateApiContractProfileError(ArbiterError):
    """Raised when a DuplicateApiContractProfileError detects a duplicate profile_id."""


class ApiContractProfileNotFoundError(ArbiterError):
    """Raised when an ApiContractProfile cannot be resolved from the registry."""


class MiddlewareConfigurationError(ConfigurationError):
    """Raised when the middleware subsystem is misconfigured."""


class MiddlewareExecutionError(ArbiterError):
    """Raised when a middleware component encounters a runtime failure."""


class DuplicateMiddlewareProfileError(ArbiterError):
    """Raised when a DuplicateMiddlewareProfileError detects a duplicate profile_id."""


class MiddlewareProfileNotFoundError(ArbiterError):
    """Raised when a MiddlewareProfile cannot be resolved from the registry."""


class InvalidLifecycleTransitionError(ArbiterError):
    """Raised when an invalid request lifecycle phase transition is attempted."""
