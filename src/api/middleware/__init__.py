"""API Middleware and Request Lifecycle Subsystem.

This subsystem provides deterministic, stateless, and observable middleware
execution pipelines for all incoming requests.
"""

from src.api.middleware.base import (
    BaseExceptionTranslator,
    BaseLifecycleManager,
    BaseMiddlewareComponent,
    Clock,
)
from src.api.middleware.correlation import CorrelationComponent
from src.api.middleware.exception_handler import (
    ExceptionTranslator,
    GlobalExceptionHandler,
)
from src.api.middleware.lifecycle import LifecycleManager
from src.api.middleware.middleware_models import (
    CorrelationContext,
    MiddlewareExecutionContext,
    MiddlewareProfile,
    RequestLifecyclePhase,
    RequestTiming,
)
from src.api.middleware.pipeline import MiddlewarePipeline
from src.api.middleware.registry import MiddlewareProfileRegistry
from src.api.middleware.timing import TimingComponent

__all__ = [
    "BaseExceptionTranslator",
    "BaseLifecycleManager",
    "BaseMiddlewareComponent",
    "Clock",
    "CorrelationComponent",
    "CorrelationContext",
    "ExceptionTranslator",
    "GlobalExceptionHandler",
    "LifecycleManager",
    "MiddlewareExecutionContext",
    "MiddlewarePipeline",
    "MiddlewareProfile",
    "MiddlewareProfileRegistry",
    "RequestLifecyclePhase",
    "RequestTiming",
    "TimingComponent",
]
