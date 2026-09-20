from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from decision.mechanics import DecisionResult as BalloonFrameDecision
from game_profiles.balloon_types import BalloonAction, TwinAction
from telemetry.outcome import SessionOutcome
from telemetry.timing import now_monotonic
from behavior.action_scheduler import ActionTimestamps, TwinActionCompletion


class InputBackend(Protocol):
    def move_to(self, x: int, y: int) -> float:
        ...

    def left_down(self) -> float:
        ...

    def left_up(self) -> float:
        ...

    def click_pair(
        self,
        first: tuple[int, int],
        second: tuple[int, int],
        inter_click_delay_s: float = 0.0,
    ) -> tuple[float, float, float, float]:
        ...


@dataclass(frozen=True)
class ExecutionContext:
    frame_id: int
    now: float
    capture_timestamp: float | None
    region_left: int
    region_top: int
    input_enabled: bool
    duration_s: float | None
    current_mouse_position: tuple[int, int]


@dataclass(frozen=True)
class PlannedInteraction:
    action: BalloonAction
    intended_x: float
    intended_y: float
    dispatch_x: int
    dispatch_y: int
    dispatch_timestamp_s: float | None = None


@dataclass(frozen=True)
class TwinPlannedInteraction:
    action: TwinAction
    first: PlannedInteraction
    second: PlannedInteraction


@dataclass(frozen=True)
class BalloonExecutionResult:
    clicks_sent: int = 0
    input_timestamp: float | None = None
    action_timestamps: ActionTimestamps | None = None
    twin_completion: TwinActionCompletion | None = None
    g_click_completed: bool = False
    completed_primary_action: BalloonAction | None = None
    outcomes: tuple[SessionOutcome, ...] = ()


class ExecutionPolicy(Protocol):
    def tick(
        self,
        decision: BalloonFrameDecision,
        context: ExecutionContext,
        input_backend: InputBackend,
    ) -> BalloonExecutionResult:
        ...

    def shutdown(
        self,
        input_backend: InputBackend,
        reason: str,
    ) -> tuple[SessionOutcome, ...]:
        ...


def default_execution_clock() -> float:
    return now_monotonic()
