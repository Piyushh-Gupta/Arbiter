"""Unit tests for the M10.3 Verification Failure Detector."""

import pytest
from pydantic import ValidationError

from src.core.exceptions import FailureAnalysisConfigurationError
from src.core.failure_analysis.analyzer import FailureAnalyzer
from src.core.failure_analysis.failure_analysis_models import (
    FailureAnalysisDefinition,
    FailureSeverity,
    VerificationFailureAnalysisDefinition,
)
from src.core.failure_analysis.implementations import VerificationFailureAnalyzer
from src.core.retrieval.retrieval_models import EvidenceBundle, RetrievalMetadata
from src.core.verification.verification_models import (
    VerificationLabel,
    VerificationMetadata,
    VerificationResult,
)


def build_verification_result(
    label: VerificationLabel, confidence: float | None
) -> VerificationResult:
    return VerificationResult(
        label=label,
        confidence=confidence,
        evidence_bundle=EvidenceBundle(
            claim="Test",
            passages=(),
            metadata=RetrievalMetadata(strategy_id="test", top_k=5),
        ),
        metadata=VerificationMetadata(strategy_id="mock"),
    )


def test_definition_immutability_and_validation() -> None:
    # Valid
    defn = VerificationFailureAnalysisDefinition(
        min_confidence_threshold=0.8, flag_nei_verdict=False
    )

    with pytest.raises(ValidationError):
        defn.min_confidence_threshold = 0.9

    # Invalid constraints
    with pytest.raises(ValidationError):
        VerificationFailureAnalysisDefinition(min_confidence_threshold=-0.1)
    with pytest.raises(ValidationError):
        VerificationFailureAnalysisDefinition(min_confidence_threshold=1.1)

    # Missing required field
    with pytest.raises(ValidationError):
        VerificationFailureAnalysisDefinition.model_validate({})


def test_compatibility_validation() -> None:
    analyzer = VerificationFailureAnalyzer()
    valid_defn = VerificationFailureAnalysisDefinition(min_confidence_threshold=0.5)
    analyzer.validate_compatibility(valid_defn)

    class OtherDef(FailureAnalysisDefinition):
        pass

    with pytest.raises(FailureAnalysisConfigurationError):
        analyzer.validate_compatibility(OtherDef())

    with pytest.raises(FailureAnalysisConfigurationError):
        # Also check at analyze time
        analyzer.analyze(
            "Test",
            build_verification_result(VerificationLabel.SUPPORTS, 0.9),
            OtherDef(),
        )


def test_absent_confidence() -> None:
    analyzer = VerificationFailureAnalyzer()
    defn = VerificationFailureAnalysisDefinition(min_confidence_threshold=0.8)

    result = build_verification_result(VerificationLabel.SUPPORTS, None)
    fa_result = analyzer.analyze("Test", result, defn)

    assert fa_result.severity == FailureSeverity.HIGH
    assert len(fa_result.failure_flags) == 1
    assert list(fa_result.failure_flags)[0].code == "ABSENT_CONFIDENCE"


def test_low_confidence() -> None:
    analyzer = VerificationFailureAnalyzer()
    defn = VerificationFailureAnalysisDefinition(min_confidence_threshold=0.8)

    # Below threshold
    result_low = build_verification_result(VerificationLabel.SUPPORTS, 0.79)
    fa_result_low = analyzer.analyze("Test", result_low, defn)

    assert fa_result_low.severity == FailureSeverity.HIGH
    assert len(fa_result_low.failure_flags) == 1
    assert list(fa_result_low.failure_flags)[0].code == "LOW_VERIFICATION_CONFIDENCE"

    # At threshold
    result_exact = build_verification_result(VerificationLabel.SUPPORTS, 0.8)
    fa_result_exact = analyzer.analyze("Test", result_exact, defn)
    assert len(fa_result_exact.failure_flags) == 0

    # Above threshold
    result_high = build_verification_result(VerificationLabel.SUPPORTS, 0.9)
    fa_result_high = analyzer.analyze("Test", result_high, defn)
    assert len(fa_result_high.failure_flags) == 0


def test_not_enough_info_verdict() -> None:
    analyzer = VerificationFailureAnalyzer()

    # Enabled
    defn_enabled = VerificationFailureAnalysisDefinition(
        min_confidence_threshold=0.5, flag_nei_verdict=True
    )
    result_nei = build_verification_result(VerificationLabel.NOT_ENOUGH_INFO, 0.9)
    fa_result_enabled = analyzer.analyze("Test", result_nei, defn_enabled)

    assert fa_result_enabled.severity == FailureSeverity.MEDIUM
    assert len(fa_result_enabled.failure_flags) == 1
    assert list(fa_result_enabled.failure_flags)[0].code == "NOT_ENOUGH_INFO_VERDICT"

    # Disabled
    defn_disabled = VerificationFailureAnalysisDefinition(
        min_confidence_threshold=0.5, flag_nei_verdict=False
    )
    fa_result_disabled = analyzer.analyze("Test", result_nei, defn_disabled)
    assert len(fa_result_disabled.failure_flags) == 0


def test_multiple_flags_and_severity_aggregation() -> None:
    analyzer = VerificationFailureAnalyzer()
    defn = VerificationFailureAnalysisDefinition(
        min_confidence_threshold=0.8, flag_nei_verdict=True
    )

    # Absent confidence + NEI -> HIGH
    result_absent_nei = build_verification_result(
        VerificationLabel.NOT_ENOUGH_INFO, None
    )
    fa_result_absent_nei = analyzer.analyze("Test", result_absent_nei, defn)

    codes = {f.code for f in fa_result_absent_nei.failure_flags}
    assert codes == {"ABSENT_CONFIDENCE", "NOT_ENOUGH_INFO_VERDICT"}
    assert fa_result_absent_nei.severity == FailureSeverity.HIGH

    # Low confidence + NEI -> HIGH
    result_low_nei = build_verification_result(VerificationLabel.NOT_ENOUGH_INFO, 0.3)
    fa_result_low_nei = analyzer.analyze("Test", result_low_nei, defn)

    codes = {f.code for f in fa_result_low_nei.failure_flags}
    assert codes == {"LOW_VERIFICATION_CONFIDENCE", "NOT_ENOUGH_INFO_VERDICT"}
    assert fa_result_low_nei.severity == FailureSeverity.HIGH


def test_clean_result() -> None:
    analyzer = VerificationFailureAnalyzer()
    defn = VerificationFailureAnalysisDefinition(min_confidence_threshold=0.5)

    result = build_verification_result(VerificationLabel.SUPPORTS, 0.9)
    fa_result = analyzer.analyze("Test", result, defn)

    assert len(fa_result.failure_flags) == 0
    assert fa_result.severity == FailureSeverity.NONE


def test_orchestrator_equivalence() -> None:
    analyzer = VerificationFailureAnalyzer()
    orchestrator = FailureAnalyzer()
    defn = VerificationFailureAnalysisDefinition(min_confidence_threshold=0.5)

    result = build_verification_result(VerificationLabel.SUPPORTS, 0.4)

    res1 = analyzer.analyze("Test", result, defn)
    res2 = orchestrator.analyze("Test", result, defn, analyzer)

    assert res1.failure_flags == res2.failure_flags
    assert res1.severity == res2.severity
    assert res1.verification_result is res2.verification_result
    assert res1.verification_result is result
