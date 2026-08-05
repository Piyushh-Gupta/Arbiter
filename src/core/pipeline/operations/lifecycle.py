from src.core.exceptions import IllegalLifecycleTransitionError
from src.core.pipeline.operations.base import BasePipelineLifecycleManager
from src.core.pipeline.operations.operation_models import PipelineLifecycleState


class PipelineLifecycleManager(BasePipelineLifecycleManager):
    """Manages deterministic state machine transitions for the pipeline lifecycle."""

    _VALID_TRANSITIONS: dict[PipelineLifecycleState, set[PipelineLifecycleState]] = {
        PipelineLifecycleState.STOPPED: {PipelineLifecycleState.INITIALIZING},
        PipelineLifecycleState.INITIALIZING: {
            PipelineLifecycleState.RUNNING,
            PipelineLifecycleState.FAILED,
        },
        PipelineLifecycleState.RUNNING: {
            PipelineLifecycleState.SHUTTING_DOWN,
            PipelineLifecycleState.FAILED,
        },
        PipelineLifecycleState.SHUTTING_DOWN: {
            PipelineLifecycleState.STOPPED,
            PipelineLifecycleState.FAILED,
        },
        PipelineLifecycleState.FAILED: {
            PipelineLifecycleState.INITIALIZING,
            PipelineLifecycleState.STOPPED,
        },
    }

    def __init__(self) -> None:
        self._current_state: PipelineLifecycleState = PipelineLifecycleState.STOPPED

    def transition_to(self, target_state: PipelineLifecycleState) -> None:
        """Attempts to transition the pipeline to a new state deterministically."""
        allowed_states = self._VALID_TRANSITIONS.get(self._current_state, set())
        if target_state not in allowed_states:
            raise IllegalLifecycleTransitionError(
                f"Cannot transition from {self._current_state} to {target_state}"
            )
        self._current_state = target_state

    @property
    def current_state(self) -> PipelineLifecycleState:
        """Gets the current lifecycle state."""
        return self._current_state
