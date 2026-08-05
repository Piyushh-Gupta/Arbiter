"""Exception translation and global handling."""

import logging

from fastapi import Request
from fastapi.responses import JSONResponse

from src.api.contracts.error_models import ErrorEnvelope
from src.core.exceptions import ArbiterError, ConfigurationError

logger = logging.getLogger(__name__)


class ExceptionTranslator:
    """Translates domain exceptions into immutable ApiErrorResponse models."""

    def translate(
        self, exception: Exception, correlation_id: str | None = None
    ) -> ErrorEnvelope:
        """Translates an exception deterministically."""
        error_type = "internal_error"
        message = "An unexpected error occurred."
        details = None

        if isinstance(exception, ConfigurationError):
            error_type = "configuration_error"
            message = str(exception)
        elif isinstance(exception, ArbiterError):
            error_type = "domain_error"
            message = str(exception)
        else:
            logger.error(f"Unhandled exception: {exception}", exc_info=True)

        return ErrorEnvelope(
            error_code=error_type,
            message=message,
            correlation_id=correlation_id,
            details=details,
        )


class GlobalExceptionHandler:
    """FastAPI integration for global exception handling."""

    def __init__(self, translator: ExceptionTranslator) -> None:
        """Initializes the handler with an exception translator."""
        self._translator = translator

    async def handle_exception(self, request: Request, exc: Exception) -> JSONResponse:
        """Handles exceptions globally and returns a standard API response."""
        # Note: in a real implementation we would extract correlation_id from request state
        # if the lifecycle manager had populated it.
        correlation_id = getattr(request.state, "correlation_id", None)

        envelope = self._translator.translate(exc, correlation_id)

        status_code = 500
        if envelope.error_code == "configuration_error":
            status_code = 500
        elif envelope.error_code == "domain_error":
            status_code = 400

        return JSONResponse(
            status_code=status_code,
            content=envelope.model_dump(),
        )
