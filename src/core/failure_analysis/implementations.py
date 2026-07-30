"""Concrete implementations for the Failure Analysis subsystem."""

from src.core.exceptions import FailureAnalysisConfigurationError
from src.core.verification.verification_models import (
    VerificationLabel,
    VerificationResult,
)

from .failure_analysis_models import (
    ContradictionAnalysisDefinition,
    FailureAnalysisDefinition,
    FailureAnalysisResult,
    FailureFlag,
    FailureMetadata,
    FailureSeverity,
    RetrievalFailureAnalysisDefinition,
    VerificationFailureAnalysisDefinition,
)


class RetrievalFailureAnalyzer:
    """
    Analyzes retrieval defects in a verification pipeline.
    Stateless and algorithmic; requires no external dependencies.
    """

    def validate_compatibility(self, definition: FailureAnalysisDefinition) -> None:
        """Validates that the definition is a RetrievalFailureAnalysisDefinition."""
        if not isinstance(definition, RetrievalFailureAnalysisDefinition):
            raise FailureAnalysisConfigurationError(
                f"RetrievalFailureAnalyzer requires RetrievalFailureAnalysisDefinition, got {type(definition).__name__}"
            )

    def analyze(
        self,
        claim: str,
        verification_result: VerificationResult,
        definition: FailureAnalysisDefinition,
    ) -> FailureAnalysisResult:
        """
        Executes structural retrieval failure analysis.
        """
        if not isinstance(definition, RetrievalFailureAnalysisDefinition):
            raise FailureAnalysisConfigurationError(
                f"RetrievalFailureAnalyzer requires RetrievalFailureAnalysisDefinition, got {type(definition).__name__}"
            )

        bundle = verification_result.evidence_bundle
        passages = bundle.passages if bundle is not None else ()

        flags: list[tuple[FailureFlag, FailureSeverity]] = []

        if len(passages) == 0:
            flags.append(
                (
                    FailureFlag(
                        code="EMPTY_BUNDLE",
                        description="The retriever returned no evidence passages.",
                    ),
                    FailureSeverity.CRITICAL,
                )
            )
            return self._build_result(flags, verification_result)

        if len(passages) < definition.min_passages:
            flags.append(
                (
                    FailureFlag(
                        code="INSUFFICIENT_EVIDENCE",
                        description=f"Retrieved {len(passages)} passages, below minimum {definition.min_passages}.",
                    ),
                    FailureSeverity.HIGH,
                )
            )

        if definition.min_score_threshold is not None:
            if all(p.score < definition.min_score_threshold for p in passages):
                flags.append(
                    (
                        FailureFlag(
                            code="LOW_RETRIEVAL_SCORES",
                            description=f"All passages scored below the minimum threshold of {definition.min_score_threshold}.",
                        ),
                        FailureSeverity.MEDIUM,
                    )
                )

        unique_identities = {(p.document_id, p.span_id) for p in passages}
        if len(unique_identities) < len(passages):
            flags.append(
                (
                    FailureFlag(
                        code="DUPLICATE_EVIDENCE",
                        description="The passage set contains duplicated corpus entries (identical document_id and span_id).",
                    ),
                    FailureSeverity.LOW,
                )
            )

        if definition.min_unique_documents is not None:
            unique_documents = {p.document_id for p in passages}
            if len(unique_documents) < definition.min_unique_documents:
                flags.append(
                    (
                        FailureFlag(
                            code="LOW_EVIDENCE_DIVERSITY",
                            description=f"Passages originated from {len(unique_documents)} unique documents, below minimum {definition.min_unique_documents}.",
                        ),
                        FailureSeverity.LOW,
                    )
                )

        return self._build_result(flags, verification_result)

    def _build_result(
        self,
        flags: list[tuple[FailureFlag, FailureSeverity]],
        verification_result: VerificationResult,
    ) -> FailureAnalysisResult:
        """Aggregates severities and constructs the final immutable result."""
        severity_order = {
            FailureSeverity.CRITICAL: 4,
            FailureSeverity.HIGH: 3,
            FailureSeverity.MEDIUM: 2,
            FailureSeverity.LOW: 1,
            FailureSeverity.NONE: 0,
        }

        final_severity = FailureSeverity.NONE
        for _, severity in flags:
            if severity_order[severity] > severity_order[final_severity]:
                final_severity = severity

        failure_flags = frozenset(flag for flag, _ in flags)

        return FailureAnalysisResult(
            failure_flags=failure_flags,
            severity=final_severity,
            verification_result=verification_result,
            metadata=FailureMetadata(strategy_id="retrieval_failure_analyzer"),
        )


class VerificationFailureAnalyzer:
    """
    Analyzes verification defects in a verification pipeline.
    Stateless and algorithmic; requires no external dependencies.
    """

    def validate_compatibility(self, definition: FailureAnalysisDefinition) -> None:
        """Validates that the definition is a VerificationFailureAnalysisDefinition."""
        if not isinstance(definition, VerificationFailureAnalysisDefinition):
            raise FailureAnalysisConfigurationError(
                f"VerificationFailureAnalyzer requires VerificationFailureAnalysisDefinition, got {type(definition).__name__}"
            )

    def analyze(
        self,
        claim: str,
        verification_result: VerificationResult,
        definition: FailureAnalysisDefinition,
    ) -> FailureAnalysisResult:
        """
        Executes structural verification failure analysis.
        """
        if not isinstance(definition, VerificationFailureAnalysisDefinition):
            raise FailureAnalysisConfigurationError(
                f"VerificationFailureAnalyzer requires VerificationFailureAnalysisDefinition, got {type(definition).__name__}"
            )

        flags: list[tuple[FailureFlag, FailureSeverity]] = []

        if verification_result.confidence is None:
            flags.append(
                (
                    FailureFlag(
                        code="ABSENT_CONFIDENCE",
                        description="The verifier did not produce a confidence score.",
                    ),
                    FailureSeverity.HIGH,
                )
            )
        elif verification_result.confidence < definition.min_confidence_threshold:
            flags.append(
                (
                    FailureFlag(
                        code="LOW_VERIFICATION_CONFIDENCE",
                        description=f"Confidence score {verification_result.confidence} is below the minimum threshold {definition.min_confidence_threshold}.",
                    ),
                    FailureSeverity.HIGH,
                )
            )

        if (
            definition.flag_nei_verdict
            and verification_result.label == VerificationLabel.NOT_ENOUGH_INFO
        ):
            flags.append(
                (
                    FailureFlag(
                        code="NOT_ENOUGH_INFO_VERDICT",
                        description="The verifier produced a NOT_ENOUGH_INFO verdict.",
                    ),
                    FailureSeverity.MEDIUM,
                )
            )

        return self._build_result(flags, verification_result)

    def _build_result(
        self,
        flags: list[tuple[FailureFlag, FailureSeverity]],
        verification_result: VerificationResult,
    ) -> FailureAnalysisResult:
        """Aggregates severities and constructs the final immutable result."""
        severity_order = {
            FailureSeverity.CRITICAL: 4,
            FailureSeverity.HIGH: 3,
            FailureSeverity.MEDIUM: 2,
            FailureSeverity.LOW: 1,
            FailureSeverity.NONE: 0,
        }

        final_severity = FailureSeverity.NONE
        for _, severity in flags:
            if severity_order[severity] > severity_order[final_severity]:
                final_severity = severity

        failure_flags = frozenset(flag for flag, _ in flags)

        return FailureAnalysisResult(
            failure_flags=failure_flags,
            severity=final_severity,
            verification_result=verification_result,
            metadata=FailureMetadata(strategy_id="verification_failure_analyzer"),
        )


class ContradictionAnalyzer:
    """
    Analyzes verified_passages for evidence contradiction.
    Stateless and algorithmic; requires no external dependencies.
    """

    def validate_compatibility(self, definition: FailureAnalysisDefinition) -> None:
        """Validates that the definition is a ContradictionAnalysisDefinition."""
        if not isinstance(definition, ContradictionAnalysisDefinition):
            raise FailureAnalysisConfigurationError(
                f"ContradictionAnalyzer requires ContradictionAnalysisDefinition, got {type(definition).__name__}"
            )

    def analyze(
        self,
        claim: str,
        verification_result: VerificationResult,
        definition: FailureAnalysisDefinition,
    ) -> FailureAnalysisResult:
        """
        Executes contradiction failure analysis.
        """
        if not isinstance(definition, ContradictionAnalysisDefinition):
            raise FailureAnalysisConfigurationError(
                f"ContradictionAnalyzer requires ContradictionAnalysisDefinition, got {type(definition).__name__}"
            )

        if verification_result.verified_passages is None:
            return FailureAnalysisResult(
                failure_flags=frozenset(),
                severity=FailureSeverity.NONE,
                verification_result=verification_result,
                metadata=FailureMetadata(strategy_id="contradiction_analyzer"),
            )

        n_supporting = 0
        n_refuting = 0

        for verified_passage in verification_result.verified_passages:
            if (
                verified_passage.label == VerificationLabel.SUPPORTS
                and verified_passage.supports_score
                >= definition.min_passage_label_confidence
            ):
                n_supporting += 1
            elif (
                verified_passage.label == VerificationLabel.REFUTES
                and verified_passage.refutes_score
                >= definition.min_passage_label_confidence
            ):
                n_refuting += 1

        flags: list[tuple[FailureFlag, FailureSeverity]] = []

        if n_supporting > 0 and n_refuting > 0:
            flags.append(
                (
                    FailureFlag(
                        code="CONTRADICTORY_EVIDENCE",
                        description="Supporting and refuting passages both present above confidence threshold.",
                    ),
                    FailureSeverity.MEDIUM,
                )
            )

        final_severity = FailureSeverity.NONE
        if flags:
            # We know it's just one flag in this current version, but keeping it general
            final_severity = FailureSeverity.MEDIUM

        failure_flags = frozenset(flag for flag, _ in flags)

        return FailureAnalysisResult(
            failure_flags=failure_flags,
            severity=final_severity,
            verification_result=verification_result,
            metadata=FailureMetadata(strategy_id="contradiction_analyzer"),
        )
