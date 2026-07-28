"""Unit tests for the M15.3 Composition Root (Bootstrap)."""

import pytest

from src.core.bootstrap import (
    build_decision_registry,
    build_evaluation_registry,
    build_explanation_registry,
    build_failure_analysis_registry,
    build_pipeline,
    build_retrieval_registry,
    build_uncertainty_registry,
    build_verification_registry,
)
from src.core.config import Settings
from src.core.pipeline.orchestrator import ArbiterPipeline


@pytest.fixture
def config() -> Settings:
    return Settings()


def test_build_retrieval_registry(config: Settings) -> None:
    registry = build_retrieval_registry(config)
    assert len(registry.profiles) == 2
    assert registry.profiles[0].profile_id == "bm25_retrieval"
    assert registry.profiles[1].profile_id == "dense_retrieval"


def test_build_verification_registry(config: Settings) -> None:
    registry = build_verification_registry(config)
    assert len(registry.profiles) == 1
    assert registry.profiles[0].profile_id == "default_verification"


def test_build_failure_analysis_registry(config: Settings) -> None:
    registry = build_failure_analysis_registry(config)
    assert len(registry.profiles) == 1
    assert registry.profiles[0].profile_id == "default_failure_analysis"


def test_build_uncertainty_registry(config: Settings) -> None:
    registry = build_uncertainty_registry(config)
    assert len(registry.profiles) == 1
    assert registry.profiles[0].profile_id == "default_uncertainty"


def test_build_decision_registry(config: Settings) -> None:
    registry = build_decision_registry(config)
    assert len(registry.profiles) == 1
    assert registry.profiles[0].profile_id == "default_decision"


def test_build_explanation_registry(config: Settings) -> None:
    registry = build_explanation_registry(config)
    assert len(registry.profiles) == 1
    assert registry.profiles[0].profile_id == "default_explanation"


def test_build_evaluation_registry(config: Settings) -> None:
    registry = build_evaluation_registry(config)
    assert len(registry.profiles) == 1
    assert registry.profiles[0].profile_id == "default_evaluation"


def test_build_pipeline(config: Settings) -> None:
    pipeline = build_pipeline(config)
    assert isinstance(pipeline, ArbiterPipeline)
    assert pipeline._retrieval_registry is not None
    assert pipeline._evaluation_registry is not None
