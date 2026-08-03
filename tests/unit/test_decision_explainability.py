"""Unit and integration tests for M4.6 Decision Explainability & Audit Reporting."""

import json
from typing import Any

import pytest

from src.core.bootstrap import build_decision_explanation_registry
from src.core.config import Settings
from src.core.decision import (
    DecisionContext,
    DecisionDefinition,
    DecisionInput,
    DecisionPolicyEngine,
)
from src.core.decision.explainability import (
    CompositeExplanationStrategy,
    DecisionExplanation,
    DecisionExplanationDefinition,
    DecisionExplanationProfile,
    DecisionExplanationProfileRegistry,
    DecisionExplanationResult,
    JsonDecisionRenderer,
    MarkdownDecisionRenderer,
    SummaryExplanationStrategy,
    TextDecisionRenderer,
    TraceAuditExplanationStrategy,
)
from src.core.exceptions import (
    DecisionExplanationProfileNotFoundError,
    DuplicateDecisionExplanationProfileError,
)


class MockVerificationResult:
    def __init__(self, confidence: float) -> None:
        self.confidence = confidence


class MockSeverityResult:
    def __init__(
        self, overall_severity: str, escalation_required: bool = False
    ) -> None:
        self.overall_severity = overall_severity
        self.escalation_required = escalation_required


# --- Registry Tests ---


def test_explanation_registry_duplicate_ids_raises() -> None:
    defn = DecisionExplanationDefinition()
    strategy = SummaryExplanationStrategy()
    p1 = DecisionExplanationProfile(profile_id="p1", definition=defn, strategy=strategy)
    p2 = DecisionExplanationProfile(profile_id="p1", definition=defn, strategy=strategy)

    with pytest.raises(DuplicateDecisionExplanationProfileError) as exc_info:
        DecisionExplanationProfileRegistry(profiles=(p1, p2))
    assert "Duplicate explanation profile ID detected: p1" in str(exc_info.value)


def test_explanation_registry_lookup_success_and_failure() -> None:
    defn = DecisionExplanationDefinition()
    strategy = SummaryExplanationStrategy()
    p1 = DecisionExplanationProfile(profile_id="p1", definition=defn, strategy=strategy)
    registry = DecisionExplanationProfileRegistry(profiles=(p1,))

    # Success lookup
    assert registry.resolve("p1") is p1

    # Failed lookup
    with pytest.raises(DecisionExplanationProfileNotFoundError) as exc_info:
        registry.resolve("missing_profile")
    assert "Decision explanation profile not found: missing_profile" in str(
        exc_info.value
    )


# --- Strategies & Deterministic Trace IDs Tests ---


@pytest.fixture
def sample_execution_context() -> Any:
    # Build a real execution context by running policy engine
    context = DecisionContext(
        verification_result=MockVerificationResult(confidence=0.96),
        severity_result=MockSeverityResult(overall_severity="MEDIUM"),
    )
    definition = DecisionDefinition(
        confidence_policy="raw", failure_policy="severity_aware"
    )
    input_data = DecisionInput(context=context, definition=definition)

    engine = DecisionPolicyEngine()
    return engine.evaluate(input_data)


def test_summary_explanation_strategy(sample_execution_context: Any) -> None:
    strategy = SummaryExplanationStrategy()
    assert strategy.strategy_id == "summary"

    definition = DecisionExplanationDefinition()
    explanation = strategy.generate_explanation(sample_execution_context, definition)

    assert isinstance(explanation, DecisionExplanation)
    assert explanation.summary["selected_action"] == "ACCEPT"
    assert explanation.summary["confidence"] == pytest.approx(0.81)  # 0.96 - 0.15
    assert explanation.summary["uncertainty"] == pytest.approx(0.19)  # 0.04 + 0.15

    # Trace ID verification
    assert "trace_id" in explanation.metadata
    assert len(explanation.metadata["trace_id"]) == 16


def test_trace_audit_explanation_strategy(sample_execution_context: Any) -> None:
    strategy = TraceAuditExplanationStrategy()
    assert strategy.strategy_id == "trace_audit"

    definition = DecisionExplanationDefinition()
    explanation = strategy.generate_explanation(sample_execution_context, definition)

    assert isinstance(explanation, DecisionExplanation)

    # Rules trace verification
    assert len(explanation.rule_trace) > 0
    rule_0 = explanation.rule_trace[0]
    assert "rule_id" in rule_0
    assert "matched" in rule_0
    assert "trace_id" in rule_0
    assert len(rule_0["trace_id"]) == 16

    # Risk trace verification
    assert len(explanation.risk_trace) == 1
    risk_0 = explanation.risk_trace[0]
    assert risk_0["factor_id"] == "severity_penalty"
    assert risk_0["confidence_delta"] == -0.15
    assert risk_0["uncertainty_delta"] == 0.15
    assert len(risk_0["trace_id"]) == 16

    # Decision trace verification
    d_trace = explanation.decision_trace
    assert "evaluated_rules" in d_trace
    assert "rejected_rules" in d_trace
    assert d_trace["selected_rule"] == "rule_accept_high_confidence"
    assert len(d_trace["trace_id"]) == 16


def test_composite_explanation_strategy(sample_execution_context: Any) -> None:
    strategy = CompositeExplanationStrategy()
    assert strategy.strategy_id == "composite"

    definition = DecisionExplanationDefinition()
    explanation = strategy.generate_explanation(sample_execution_context, definition)

    assert isinstance(explanation, DecisionExplanation)
    assert explanation.summary["selected_action"] == "ACCEPT"
    assert len(explanation.rule_trace) > 0
    assert len(explanation.risk_trace) == 1
    assert "trace_id" in explanation.metadata
    assert len(explanation.metadata["trace_id"]) == 16


# --- Format Renderers Tests ---


def test_markdown_renderer(sample_execution_context: Any) -> None:
    strategy = CompositeExplanationStrategy()
    definition = DecisionExplanationDefinition()
    explanation = strategy.generate_explanation(sample_execution_context, definition)

    renderer = MarkdownDecisionRenderer()
    assert renderer.renderer_id == "markdown"

    rendered_text = renderer.render(explanation)
    assert "# Decision Audit Explanation Report" in rendered_text
    assert "## Decision Summary" in rendered_text
    assert "## Evaluated Decision Rules" in rendered_text
    assert "## Operational Risk Adjustments" in rendered_text
    assert "## Decision Execution Trace" in rendered_text
    assert "rule_accept_high_confidence" in rendered_text


def test_json_renderer(sample_execution_context: Any) -> None:
    strategy = CompositeExplanationStrategy()
    definition = DecisionExplanationDefinition()
    explanation = strategy.generate_explanation(sample_execution_context, definition)

    renderer = JsonDecisionRenderer()
    assert renderer.renderer_id == "json"

    rendered_json = renderer.render(explanation)
    parsed = json.loads(rendered_json)

    assert "summary" in parsed
    assert "rule_trace" in parsed
    assert "risk_trace" in parsed
    assert "decision_trace" in parsed
    assert parsed["summary"]["selected_action"] == "ACCEPT"


def test_text_renderer(sample_execution_context: Any) -> None:
    strategy = CompositeExplanationStrategy()
    definition = DecisionExplanationDefinition()
    explanation = strategy.generate_explanation(sample_execution_context, definition)

    renderer = TextDecisionRenderer()
    assert renderer.renderer_id == "text"

    rendered_text = renderer.render(explanation)
    assert "DECISION AUDIT REPORT" in rendered_text
    assert "SUMMARY:" in rendered_text
    assert "RULES EVALUATED:" in rendered_text
    assert "RISK ADJUSTMENTS:" in rendered_text
    assert "EXECUTION TRACE:" in rendered_text


# --- Bootstrap Registry Construction Tests ---


def test_bootstrap_builds_explanation_registries_correctly() -> None:
    config = Settings()
    registry = build_decision_explanation_registry(config)

    assert isinstance(registry, DecisionExplanationProfileRegistry)
    profile = registry.resolve("default_decision_explanation")
    assert profile.profile_id == "default_decision_explanation"
    assert isinstance(profile.strategy, CompositeExplanationStrategy)


# --- End-to-End Integration Flow Tests ---


def test_decision_explainability_integration_pipeline(
    sample_execution_context: Any,
) -> None:
    # 1. Define formatting scope
    definition = DecisionExplanationDefinition(template_format="json")

    # 2. Strategy produces structured explanation
    strategy = CompositeExplanationStrategy()
    explanation = strategy.generate_explanation(sample_execution_context, definition)
    assert isinstance(explanation, DecisionExplanation)

    # 3. Renderer formats report
    renderer = JsonDecisionRenderer()
    rendered_content = renderer.render(explanation)
    assert "rule_accept_high_confidence" in rendered_content

    # 4. Result encapsulates result
    result = DecisionExplanationResult(
        explanation=explanation,
        rendered_format=rendered_content,
        renderer_id=renderer.renderer_id,
    )

    assert result.rendered_format == rendered_content
    assert result.renderer_id == "json"
