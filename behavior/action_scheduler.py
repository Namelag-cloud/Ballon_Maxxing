from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Optional, Protocol

from behavior.model import BehavioralAction
from telemetry.timing import now_monotonic


class ActionPhase(Enum):
    IDLE = auto()
    WAITING_REACTION = auto()
    MOVING = auto()
    CLICK_DOWN = auto()
    CLICK_HOLD = auto()
    COMPLETE = auto()


@dataclass
class ActionTimestamps:
    action_planned: float = 0.0
    reaction_start: float = 0.0
    reaction_end: float = 0.0
    movement_start: float = 0.0
    movement_end: float = 0.0
    click_down: float = 0.0
    click_up: float = 0.0
    action_complete: float = 0.0
    interrupted: bool = False
    cancel_reason: str | None = None


@dataclass
class ActiveAction:
    behavioral_action: BehavioralAction
    target_x: int
    target_y: int
    phase: ActionPhase = ActionPhase.WAITING_REACTION
    timestamps: ActionTimestamps = field(default_factory=ActionTimestamps)
    trajectory_index: int = 0
    intended_target_x: int | None = None
    intended_target_y: int | None = None
    _clock: callable = field(default_factory=lambda: now_monotonic, repr=False)

    @property
    def reaction_time_s(self) -> float:
        return self.behavioral_action.reaction_time_s

    @property
    def movement_time_s(self) -> float:
        return self.behavioral_action.movement_time_s

    @property
    def click_duration_s(self) -> float:
        return self.behavioral_action.click_duration_s

    @property
    def trajectory(self):
        return self.behavioral_action.trajectory

    @property
    def planned_total_s(self) -> float:
        return self.behavioral_action.total_time_s


class InputBackend(Protocol):
    def move_to(self, x: int, y: int) -> float: ...
    def left_down(self) -> float: ...
    def left_up(self) -> float: ...


class ActionScheduler:
    def __init__(self, clock=None, diagnostic: Any = None) -> None:
        self._clock = clock or now_monotonic
        self._diagnostic = diagnostic
        self._active: Optional[ActiveAction] = None
        self._completed_history: list[ActionTimestamps] = []

    @property
    def is_idle(self) -> bool:
        return self._active is None or self._active.phase == ActionPhase.COMPLETE

    @property
    def active_action(self) -> Optional[ActiveAction]:
        return self._active

    @property
    def completed_history(self) -> list[ActionTimestamps]:
        return self._completed_history

    def start_action(self, behavioral_action: BehavioralAction, target_x: int, target_y: int) -> None:
        if not self.is_idle:
            if self._diagnostic:
                self._diagnostic.log("scheduler_start_blocked", scheduler="primary", phase=self._active.phase.name if self._active else None)
            return
        now = self._clock()
        action = ActiveAction(
            behavioral_action=behavioral_action,
            target_x=int(round(behavioral_action.click_position[0])),
            target_y=int(round(behavioral_action.click_position[1])),
            phase=ActionPhase.WAITING_REACTION,
            intended_target_x=int(target_x),
            intended_target_y=int(target_y),
            _clock=self._clock,
        )
        # Keep the explicit arguments for backwards-compatible call sites, but
        # the behavioral endpoint is now authoritative for actual OS input.
        if behavioral_action.click_position is None:
            action.target_x, action.target_y = target_x, target_y
        action.timestamps.action_planned = now
        action.timestamps.reaction_start = now
        self._active = action
        if self._diagnostic:
            self._diagnostic.log("scheduler_start", scheduler="primary", phase=action.phase.name, target_x=action.target_x, target_y=action.target_y, intended_target_x=action.intended_target_x, intended_target_y=action.intended_target_y)

    def tick(self, input_backend: InputBackend) -> Optional[ActionTimestamps]:
        if self._active is None:
            return None
        action = self._active
        now = self._clock()

        if action.phase == ActionPhase.WAITING_REACTION:
            if now - action.timestamps.reaction_start >= action.reaction_time_s:
                action.timestamps.reaction_end = now
                action.timestamps.movement_start = now
                action.phase = ActionPhase.MOVING
                action.trajectory_index = 0
                if self._diagnostic:
                    self._diagnostic.log("scheduler_phase", scheduler="primary", phase=action.phase.name)
                return None

        if action.phase == ActionPhase.MOVING:
            if not action.trajectory:
                action.timestamps.movement_end = now
                input_backend.move_to(action.target_x, action.target_y)
                action.phase = ActionPhase.CLICK_DOWN
                if self._diagnostic:
                    self._diagnostic.log("scheduler_phase", scheduler="primary", phase=action.phase.name)
            else:
                self._tick_trajectory(action, input_backend, now)
            if action.phase == ActionPhase.CLICK_DOWN:
                return None

        if action.phase == ActionPhase.CLICK_DOWN:
            action.timestamps.click_down = input_backend.left_down()
            action.phase = ActionPhase.CLICK_HOLD
            if self._diagnostic:
                self._diagnostic.log("scheduler_phase", scheduler="primary", phase=action.phase.name)

        if action.phase == ActionPhase.CLICK_HOLD:
            if self._clock() - action.timestamps.click_down >= action.click_duration_s:
                action.timestamps.click_up = input_backend.left_up()
                action.timestamps.action_complete = self._clock()
                action.phase = ActionPhase.COMPLETE
                self._completed_history.append(action.timestamps)
                if self._diagnostic:
                    self._diagnostic.log("scheduler_phase", scheduler="primary", phase=action.phase.name)
                return action.timestamps
        return None

    def _tick_trajectory(self, action: ActiveAction, input_backend: InputBackend, now: float) -> None:
        trajectory = action.trajectory
        while action.trajectory_index < len(trajectory):
            pt = trajectory[action.trajectory_index]
            if now < action.timestamps.movement_start + pt.t:
                break
            input_backend.move_to(int(pt.x), int(pt.y))
            action.trajectory_index += 1
        if action.trajectory_index >= len(trajectory):
            # Do not restore the target center: click_position already contains
            # the modeled endpoint error/miss endpoint.
            input_backend.move_to(action.target_x, action.target_y)
            action.timestamps.movement_end = now
            action.phase = ActionPhase.CLICK_DOWN

    def cancel(self, input_backend: InputBackend | None = None, reason: str = "cancelled") -> Optional[ActionTimestamps]:
        if self._active is None:
            return None
        action = self._active
        action.timestamps.interrupted = True
        action.timestamps.cancel_reason = reason
        if action.phase == ActionPhase.CLICK_HOLD and input_backend is not None:
            try:
                action.timestamps.click_up = input_backend.left_up()
            except Exception as exc:
                if self._diagnostic:
                    self._diagnostic.log(
                        "sendinput_dispatch_error",
                        kind="shutdown_mouse_up",
                        error=str(exc),
                    )
        action.timestamps.action_complete = self._clock()
        action.phase = ActionPhase.COMPLETE
        self._completed_history.append(action.timestamps)
        timestamps = action.timestamps
        self._active = None
        if self._diagnostic:
            self._diagnostic.log(
                "scheduler_cancel",
                scheduler="primary",
                reason=reason,
                phase=action.phase.name,
                interrupted=True,
            )
        return timestamps

    def force_complete(self) -> Optional[ActionTimestamps]:
        return self.cancel()

    def clear(self) -> None:
        self._active = None
        if self._diagnostic:
            self._diagnostic.log("scheduler_clear", scheduler="primary")


class PairInputBackend(Protocol):
    def click_pair(
        self,
        first: tuple[int, int],
        second: tuple[int, int],
        inter_click_delay_s: float = 0.0,
    ) -> tuple[float, float, float, float]:
        """Dispatch two complete clicks and return individual timestamps."""


@dataclass(frozen=True)
class TwinActionCompletion:
    first: ActionTimestamps
    second: ActionTimestamps
    click_down_delta_ms: float
    intended_click_down_delta_ms: float = 0.0


class TwinActionScheduler:
    """Coordinate two behavioral plans while preserving a measurable delay."""

    def __init__(self, clock=None, diagnostic: Any = None) -> None:
        self._clock = clock or now_monotonic
        self._diagnostic = diagnostic
        self._active: tuple[ActiveAction, ActiveAction] | None = None
        self._max_click_delta_ms = 500.0
        self._inter_click_delay_s = 0.0

    @property
    def is_idle(self) -> bool:
        return self._active is None or all(a.phase == ActionPhase.COMPLETE for a in self._active)

    @property
    def active_actions(self) -> tuple[ActiveAction, ActiveAction] | None:
        return self._active

    def start_action(
        self, first_behavior: BehavioralAction, first_x: int, first_y: int,
        second_behavior: BehavioralAction, second_x: int, second_y: int,
        *, max_click_delta_ms: float = 500.0,
        inter_click_delay_s: float = 0.0,
    ) -> None:
        if not self.is_idle:
            if self._diagnostic:
                self._diagnostic.log("scheduler_start_blocked", scheduler="twin", phase=[a.phase.name for a in self._active] if self._active else None)
            return
        now = self._clock()
        first = ActiveAction(
            first_behavior, int(round(first_behavior.click_position[0])),
            int(round(first_behavior.click_position[1])),
            intended_target_x=int(first_x), intended_target_y=int(first_y),
            _clock=self._clock
        )
        second = ActiveAction(
            second_behavior, int(round(second_behavior.click_position[0])),
            int(round(second_behavior.click_position[1])),
            intended_target_x=int(second_x), intended_target_y=int(second_y),
            _clock=self._clock
        )
        first.timestamps.action_planned = second.timestamps.action_planned = now
        first.timestamps.reaction_start = second.timestamps.reaction_start = now
        self._max_click_delta_ms = max_click_delta_ms
        self._inter_click_delay_s = max(0.0, inter_click_delay_s)
        self._active = (first, second)
        if self._diagnostic:
            self._diagnostic.log("scheduler_start", scheduler="twin", phase=[first.phase.name, second.phase.name], targets=[[first.target_x, first.target_y], [second.target_x, second.target_y]])

    def tick(self, input_backend: PairInputBackend) -> TwinActionCompletion | None:
        if self._active is None:
            return None
        first, second = self._active
        now = self._clock()
        self._advance_reaction(first, now)
        self._advance_reaction(second, now)
        if first.phase == ActionPhase.MOVING:
            self._advance_movement(first, input_backend, now)
        if second.phase == ActionPhase.MOVING:
            self._advance_movement(second, input_backend, now)
        if first.phase != ActionPhase.CLICK_DOWN or second.phase != ActionPhase.CLICK_DOWN:
            return None
        if not hasattr(input_backend, "click_pair"):
            raise TypeError("TwinActionScheduler requires InputBackend.click_pair")

        try:
            first_down, first_up, second_down, second_up = input_backend.click_pair(
                (first.target_x, first.target_y),
                (second.target_x, second.target_y),
                self._inter_click_delay_s,
            )
        except TypeError:
            # Preserve compatibility with test/dummy backends implementing the
            # pre-profile two-argument contract.
            first_down, first_up, second_down, second_up = input_backend.click_pair(
                (first.target_x, first.target_y),
                (second.target_x, second.target_y),
            )
        delta_ms = abs(first_down - second_down) * 1000.0
        if delta_ms > self._max_click_delta_ms:
            raise RuntimeError(
                f"Twin click synchronization exceeded {self._max_click_delta_ms:.1f} ms: {delta_ms:.3f} ms"
            )
        first.timestamps.click_down = first_down
        first.timestamps.click_up = first_up
        first.timestamps.action_complete = first_up
        second.timestamps.click_down = second_down
        second.timestamps.click_up = second_up
        second.timestamps.action_complete = second_up
        first.phase = ActionPhase.COMPLETE
        second.phase = ActionPhase.COMPLETE
        if self._diagnostic:
            self._diagnostic.log("scheduler_phase", scheduler="twin", phase=[first.phase.name, second.phase.name], click_down_delta_ms=delta_ms)
        return TwinActionCompletion(
            first.timestamps, second.timestamps, delta_ms,
            self._inter_click_delay_s * 1000.0,
        )

    def clear(self) -> None:
        self._active = None
        if self._diagnostic:
            self._diagnostic.log("scheduler_clear", scheduler="twin")

    @staticmethod
    def _advance_reaction(action: ActiveAction, now: float) -> None:
        if action.phase == ActionPhase.WAITING_REACTION and now - action.timestamps.reaction_start >= action.reaction_time_s:
            action.timestamps.reaction_end = now
            action.timestamps.movement_start = now
            action.phase = ActionPhase.MOVING
            action.trajectory_index = 0

    @staticmethod
    def _advance_movement(action: ActiveAction, input_backend: InputBackend, now: float) -> None:
        trajectory = action.trajectory
        if not trajectory:
            action.timestamps.movement_end = now
            input_backend.move_to(action.target_x, action.target_y)
            action.phase = ActionPhase.CLICK_DOWN
            return
        while action.trajectory_index < len(trajectory):
            pt = trajectory[action.trajectory_index]
            if now < action.timestamps.movement_start + pt.t:
                break
            input_backend.move_to(int(pt.x), int(pt.y))
            action.trajectory_index += 1
        if action.trajectory_index >= len(trajectory):
            input_backend.move_to(action.target_x, action.target_y)
            action.timestamps.movement_end = now
            action.phase = ActionPhase.CLICK_DOWN
