"""Tests for ExceptionTranslator."""

from src.api.middleware.exception_handler import ExceptionTranslator
from src.core.exceptions import ArbiterError, ConfigurationError


def test_translate_configuration_error() -> None:
    translator = ExceptionTranslator()
    error = ConfigurationError("Test config error")

    envelope = translator.translate(error, correlation_id="test-corr")

    assert envelope.error_code == "configuration_error"
    assert envelope.message == "Test config error"
    assert envelope.correlation_id == "test-corr"


def test_translate_domain_error() -> None:
    translator = ExceptionTranslator()
    error = ArbiterError("Test domain error")

    envelope = translator.translate(error, correlation_id="test-corr")

    assert envelope.error_code == "domain_error"
    assert envelope.message == "Test domain error"
    assert envelope.correlation_id == "test-corr"


def test_translate_unhandled_error() -> None:
    translator = ExceptionTranslator()
    error = ValueError("Test value error")

    envelope = translator.translate(error, correlation_id="test-corr")

    assert envelope.error_code == "internal_error"
    assert envelope.message == "An unexpected error occurred."
    assert envelope.correlation_id == "test-corr"
