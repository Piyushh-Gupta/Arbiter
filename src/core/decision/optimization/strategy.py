"""Optimized decision strategy implementation (M4.7)."""

import time
from typing import Any

from src.core.decision.base import BaseDecisionStrategy
from src.core.decision.decision_models import (
    DecisionContext,
    DecisionDefinition,
    DecisionResult,
    compute_decision_fingerprint,
)
from src.core.decision.implementations import PolicyDecisionStrategy
from src.core.decision.optimization.cache import BaseDecisionCache
from src.core.decision.optimization.guard import DecisionExecutionGuard
from src.core.decision.optimization.optimization_models import DecisionExecutionMetrics


class OptimizedDecisionStrategy(BaseDecisionStrategy):
    """Resilient and cached wrapper around PolicyDecisionStrategy execution."""

    def __init__(
        self,
        cache: BaseDecisionCache,
        guard: DecisionExecutionGuard,
        delegate: PolicyDecisionStrategy | None = None,
    ) -> None:
        self._cache = cache
        self._guard = guard
        self._delegate = delegate or PolicyDecisionStrategy()

    @property
    def strategy_id(self) -> str:
        return "optimized_decision_strategy"

    def validate_compatibility(self, definition: DecisionDefinition) -> None:
        if self._delegate:
            self._delegate.validate_compatibility(definition)

    def decide(
        self,
        context_or_claim: Any,
        definition_or_unc: Any = None,
        definition: Any = None,
    ) -> DecisionResult:
        """Executes decision logic resiliently with caching, timeout, and metrics collection."""
        start_time = time.perf_counter()
        import hashlib

        if isinstance(context_or_claim, DecisionContext):
            context = context_or_claim
            effective_def = definition_or_unc or DecisionDefinition()
        else:
            context = DecisionContext(
                calibration_result=definition_or_unc,
                metadata={"claim": str(context_or_claim)},
            )
            effective_def = definition or DecisionDefinition()

        effective_def_obj = (
            effective_def
            if isinstance(effective_def, DecisionDefinition)
            else DecisionDefinition()
        )

        raw_conf = 0.0
        if context.verification_result and hasattr(
            context.verification_result, "confidence"
        ):
            raw_conf = float(getattr(context.verification_result, "confidence", 0.0))

        severity = "NONE"
        if context.severity_result and hasattr(
            context.severity_result, "overall_severity"
        ):
            severity = str(getattr(context.severity_result, "overall_severity", "NONE"))

        def_hash = compute_decision_fingerprint(effective_def_obj)
        input_repr = f"{raw_conf:.4f}_{severity}_{def_hash}"
        fingerprint = hashlib.sha256(input_repr.encode()).hexdigest()

        # 1. Cache lookup
        cached_result = self._cache.get(fingerprint)
        if cached_result is not None:
            latency_ms = (time.perf_counter() - start_time) * 1000.0

            # Build cache hit metrics
            metrics = DecisionExecutionMetrics(
                decision_latency_ms=latency_ms,
                cache_hit=True,
                fallback_used=False,
                evaluated_policy_count=0,
                total_execution_ms=latency_ms,
            )

            # Reconstruct result to embed metrics
            return DecisionResult(
                final_verdict=cached_result.final_verdict,
                final_confidence=cached_result.final_confidence,
                final_uncertainty=cached_result.final_uncertainty,
                decision_trace=cached_result.decision_trace,
                escalation_required=cached_result.escalation_required,
                metadata=cached_result.metadata,
                execution_metrics=metrics,
            )

        # 2. Guarded execution
        fallback_used = False

        def execute_delegate() -> DecisionResult:
            return self._delegate.decide(context, effective_def_obj)

        result = self._guard.execute(execute_delegate)

        # Check if fallback was triggered
        if result.metadata.strategy_id == "execution_guard_fallback":
            fallback_used = True

        latency_ms = (time.perf_counter() - start_time) * 1000.0

        # 3. Cache store on successful non-fallback outcomes
        if not fallback_used:
            self._cache.put(fingerprint, result)

        # Extract rule evaluation details
        policy_count = 0
        if result.decision_trace and hasattr(result.decision_trace, "reasoning"):
            # A rough estimate or count of rules
            policy_count = 1

        # 4. Observability Metrics
        metrics = DecisionExecutionMetrics(
            decision_latency_ms=latency_ms,
            cache_hit=False,
            fallback_used=fallback_used,
            evaluated_policy_count=policy_count,
            total_execution_ms=latency_ms,
        )

        return DecisionResult(
            final_verdict=result.final_verdict,
            final_confidence=result.final_confidence,
            final_uncertainty=result.final_uncertainty,
            decision_trace=result.decision_trace,
            escalation_required=result.escalation_required,
            metadata=result.metadata,
            execution_metrics=metrics,
        )
