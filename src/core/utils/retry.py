"""Reusable retry policy for transient failures."""

import time
from collections.abc import Callable
from typing import Any, TypeVar

import structlog

from src.core.config import settings

logger = structlog.get_logger(__name__)

T = TypeVar("T")


def with_retry(
    exceptions: tuple[type[Exception], ...] = (Exception,),
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """
    Decorator for wrapping functions with an exponential backoff retry policy.

    Uses DownloadSettings from global configuration.
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        def wrapper(*args: Any, **kwargs: Any) -> T:
            max_retries = settings.download.max_retries
            backoff_factor = settings.download.backoff_factor

            attempt = 0
            while True:
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    if attempt >= max_retries:
                        logger.error(
                            "Max retries exhausted",
                            func=func.__name__,
                            attempt=attempt,
                            error=str(e),
                        )
                        raise

                    sleep_time = backoff_factor**attempt
                    logger.warning(
                        "Transient failure, retrying",
                        func=func.__name__,
                        attempt=attempt,
                        sleep_time=sleep_time,
                        error=str(e),
                    )
                    time.sleep(sleep_time)
                    attempt += 1

        return wrapper

    return decorator
