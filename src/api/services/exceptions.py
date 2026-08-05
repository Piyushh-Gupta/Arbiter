"""Exception translation layer."""

from fastapi import HTTPException, status
from pydantic import ValidationError

from src.core.exceptions import ArbiterError, ConfigurationError


class ExceptionTranslator:
    """Translates domain exceptions to HTTP exceptions deterministically."""

    @staticmethod
    def translate(exc: Exception) -> HTTPException:
        """Translates an exception to an HTTPException."""
        if isinstance(exc, ValidationError):
            return HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=str(exc),
            )
        if isinstance(exc, ConfigurationError):
            return HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=str(exc),
            )
        if isinstance(exc, ArbiterError):
            return HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(exc),
            )
        return HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal Server Error",
        )
