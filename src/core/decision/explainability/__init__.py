"""Decision Explainability & Audit Reporting (M4.6)."""

from src.core.decision.explainability.base import BaseDecisionExplanationStrategy
from src.core.decision.explainability.explainability_models import (
    DecisionExplanation,
    DecisionExplanationDefinition,
    DecisionExplanationProfile,
    DecisionExplanationProfileRegistry,
    DecisionExplanationResult,
)
from src.core.decision.explainability.rendering import (
    BaseDecisionRenderer,
    JsonDecisionRenderer,
    MarkdownDecisionRenderer,
    TextDecisionRenderer,
)
from src.core.decision.explainability.strategies import (
    CompositeExplanationStrategy,
    SummaryExplanationStrategy,
    TraceAuditExplanationStrategy,
)

__all__ = [
    "BaseDecisionExplanationStrategy",
    "DecisionExplanation",
    "DecisionExplanationDefinition",
    "DecisionExplanationProfile",
    "DecisionExplanationProfileRegistry",
    "DecisionExplanationResult",
    "BaseDecisionRenderer",
    "JsonDecisionRenderer",
    "MarkdownDecisionRenderer",
    "TextDecisionRenderer",
    "CompositeExplanationStrategy",
    "SummaryExplanationStrategy",
    "TraceAuditExplanationStrategy",
]
