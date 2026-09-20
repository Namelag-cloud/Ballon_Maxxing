from __future__ import annotations

from telemetry.outcome import SessionOutcome
from diagnostics.downstream import DownstreamDiagnostic

from .execution_policy import (
    BalloonExecutionResult,
    ExecutionContext,
    ExecutionPolicy,
    InputBackend,
)


class HumanizedExecutionPolicy:
    def __init__(self, diagnostic: DownstreamDiagnostic | None = None) -> None:
        self._diagnostic = diagnostic

    def tick(
        self,
        decision,
        context: ExecutionContext,
        input_backend: InputBackend,
    ) -> BalloonExecutionResult:
        raise NotImplementedError(
            "HumanizedExecutionPolicy is not implemented until Phase B"
        )

    def shutdown(
        self,
        input_backend: InputBackend,
        reason: str,
    ) -> tuple[SessionOutcome, ...]:
        raise NotImplementedError(
            "HumanizedExecutionPolicy is not implemented until Phase B"
        )
