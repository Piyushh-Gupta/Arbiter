"""Unit and integration tests for M4.7 Decision Engine Production Optimization & Hardening."""

import hashlib
import time

import pytest

from src.core.bootstrap import build_decision_optimization_registry
from src.core.config import Settings
from src.core.decision import (
    DecisionContext,
    DecisionDefinition,
    DecisionMetadata,
    DecisionResult,
    DecisionTrace,
    compute_decision_fingerprint,
)
from src.core.decision.optimization import (
    DecisionExecutionGuard,
    DecisionExecutionGuardDefinition,
    DecisionOptimizationDefinition,
    DecisionOptimizationProfile,
    DecisionOptimizationProfileRegistry,
    InMemoryDecisionCache,
    OptimizedDecisionStrategy,
)
from src.core.exceptions import (
    DecisionOptimizationProfileNotFoundError,
    DuplicateDecisionOptimizationProfileError,
)


class MockVerificationResult:
    def __init__(self, confidence: float) -> None:
        self.confidence = confidence


class MockSeverityResult:
    def __init__(self, overall_severity: str) -> None:
        self.overall_severity = overall_severity


# Helper to build a valid DecisionResult
def make_result(
    verdict: str = "ACCEPT",
    confidence: float = 0.9,
    uncertainty: float = 0.1,
    strategy_id: str = "test",
) -> DecisionResult:
    return DecisionResult(
        final_verdict=verdict,
        final_confidence=confidence,
        final_uncertainty=uncertainty,
        decision_trace=DecisionTrace(selected_rule="r1"),
        metadata=DecisionMetadata(
            strategy_id=strategy_id, configuration_fingerprint="abc"
        ),
    )


# --- Registry Tests ---


def test_optimization_registry_duplicate_ids_raises() -> None:
    defn = DecisionOptimizationDefinition()
    p1 = DecisionOptimizationProfile(profile_id="p1", definition=defn)
    p2 = DecisionOptimizationProfile(profile_id="p1", definition=defn)

    with pytest.raises(DuplicateDecisionOptimizationProfileError) as exc_info:
        DecisionOptimizationProfileRegistry(profiles=(p1, p2))
    assert "Duplicate optimization profile ID detected: p1" in str(exc_info.value)


def test_optimization_registry_lookup_success_and_failure() -> None:
    defn = DecisionOptimizationDefinition()
    p1 = DecisionOptimizationProfile(profile_id="p1", definition=defn)
    registry = DecisionOptimizationProfileRegistry(profiles=(p1,))

    # Success
    assert registry.resolve("p1") is p1

    # Failure
    with pytest.raises(DecisionOptimizationProfileNotFoundError) as exc_info:
        registry.resolve("missing_profile")
    assert "Decision optimization profile not found: missing_profile" in str(
        exc_info.value
    )


# --- Cache Tests ---


def test_in_memory_cache_hit_and_miss() -> None:
    cache = InMemoryDecisionCache(max_size=3, ttl_seconds=10)
    res = make_result()

    assert cache.get("key1") is None
    assert cache.contains("key1") is False

    cache.put("key1", res)
    assert cache.contains("key1") is True
    assert cache.get("key1") is res


def test_in_memory_cache_lru_eviction() -> None:
    cache = InMemoryDecisionCache(max_size=2, ttl_seconds=10)
    res = make_result()

    cache.put("k1", res)
    cache.put("k2", res)
    assert cache.contains("k1") is True
    assert cache.contains("k2") is True

    # Evict k1 when putting k3
    cache.put("k3", res)
    assert cache.contains("k1") is False
    assert cache.contains("k2") is True
    assert cache.contains("k3") is True


def test_in_memory_cache_ttl_expiration() -> None:
    # TTL of 0 seconds to simulate instant expiration
    cache = InMemoryDecisionCache(max_size=5, ttl_seconds=0)
    res = make_result()

    cache.put("k1", res)
    time.sleep(0.01)
    assert cache.contains("k1") is False
    assert cache.get("k1") is None


# --- Execution Guard Tests ---


def test_execution_guard_success() -> None:
    guard_def = DecisionExecutionGuardDefinition(
        timeout_ms=500, max_retries=1, fallback_action="ABSTAIN"
    )
    guard = DecisionExecutionGuard(guard_def)
    res = make_result()

    called_count = 0

    def dummy_call() -> DecisionResult:
        nonlocal called_count
        called_count += 1
        return res

    out = guard.execute(dummy_call)
    assert out.final_verdict == "ACCEPT"
    assert called_count == 1


def test_execution_guard_timeout_fallback() -> None:
    guard_def = DecisionExecutionGuardDefinition(
        timeout_ms=50, max_retries=1, fallback_action="ESCALATE"
    )
    guard = DecisionExecutionGuard(guard_def)

    def slow_call() -> DecisionResult:
        time.sleep(0.1)  # slow execution
        return make_result()

    out = guard.execute(slow_call)
    # Should fall back to ESCALATE because timeout is 50ms and execution is 100ms
    assert out.final_verdict == "ESCALATE"
    assert out.metadata.strategy_id == "execution_guard_fallback"


def test_execution_guard_retry_and_success() -> None:
    guard_def = DecisionExecutionGuardDefinition(
        timeout_ms=200, max_retries=2, fallback_action="ABSTAIN"
    )
    guard = DecisionExecutionGuard(guard_def)
    res = make_result()

    called_count = 0

    def transient_failure_call() -> DecisionResult:
        nonlocal called_count
        called_count += 1
        if called_count < 2:
            raise RuntimeError("Transient DB Error")
        return res

    out = guard.execute(transient_failure_call)
    assert out.final_verdict == "ACCEPT"
    assert called_count == 2  # succeeded on second attempt (1 retry used)


def test_execution_guard_persistent_failure_fallback() -> None:
    guard_def = DecisionExecutionGuardDefinition(
        timeout_ms=200, max_retries=2, fallback_action="ABSTAIN"
    )
    guard = DecisionExecutionGuard(guard_def)

    called_count = 0

    def persistent_failure_call() -> DecisionResult:
        nonlocal called_count
        called_count += 1
        raise RuntimeError("Persistent DB Error")

    out = guard.execute(persistent_failure_call)
    assert out.final_verdict == "ABSTAIN"
    assert out.metadata.strategy_id == "execution_guard_fallback"
    assert called_count == 3  # Initial try + 2 retries = 3 calls


# --- Bootstrap Tests ---


def test_bootstrap_builds_optimization_registries_correctly() -> None:
    config = Settings()
    registry = build_decision_optimization_registry(config)

    assert isinstance(registry, DecisionOptimizationProfileRegistry)
    profile = registry.resolve("default_decision_optimization")
    assert profile.profile_id == "default_decision_optimization"
    assert profile.definition.cache_config.enabled is True
    assert profile.definition.guard_config.fallback_action == "ABSTAIN"


# --- Integration pipeline Tests ---


def test_optimized_decision_strategy_integration_flow() -> None:
    cache = InMemoryDecisionCache()
    guard_def = DecisionExecutionGuardDefinition(
        timeout_ms=200, max_retries=1, fallback_action="ABSTAIN"
    )
    guard = DecisionExecutionGuard(guard_def)

    strategy = OptimizedDecisionStrategy(cache=cache, guard=guard)

    context = DecisionContext(
        verification_result=MockVerificationResult(confidence=0.96),
        severity_result=MockSeverityResult(overall_severity="MEDIUM"),
    )
    definition = DecisionDefinition(confidence_policy="raw")

    # Fingerprint check logic matches OptimizedDecisionStrategy:
    raw_conf = 0.96
    severity = "MEDIUM"
    def_hash = compute_decision_fingerprint(definition)
    input_repr = f"{raw_conf:.4f}_{severity}_{def_hash}"
    fingerprint = hashlib.sha256(input_repr.encode()).hexdigest()

    # 1. First execution - Cache Miss
    res1 = strategy.decide(context, definition)
    assert res1.final_verdict == "ACCEPT"
    assert res1.execution_metrics is not None

    metrics1 = res1.execution_metrics
    assert metrics1.cache_hit is False
    assert metrics1.fallback_used is False
    assert metrics1.decision_latency_ms > 0.0

    # Verify placed in cache
    assert cache.contains(fingerprint) is True

    # 2. Second execution - Cache Hit
    res2 = strategy.decide(context, definition)
    assert res2.final_verdict == "ACCEPT"

    assert res2.execution_metrics is not None
    metrics2 = res2.execution_metrics
    assert metrics2.cache_hit is True
    assert metrics2.fallback_used is False
