from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol

from behavior.action_scheduler import ActionScheduler, ActionTimestamps, TwinActionCompletion, TwinActionScheduler
from behavior.model import BehavioralAction, BehavioralModel
from game_profiles.balloon_types import BalloonAction, TwinAction
from input.coordinates import frame_to_screen


class PairInputBackend(Protocol):
    def click_pair(
        self,
        first: tuple[int, int],
        second: tuple[int, int],
        inter_click_delay_s: float = 0.0,
    ) -> tuple[float, float, float, float]:
        ...


@dataclass(frozen=True)
class BalloonExecutionResult:
    clicks_sent: int = 0
    input_timestamp: Optional[float] = None
    action_timestamps: Optional[ActionTimestamps] = None
    twin_completion: Optional[TwinActionCompletion] = None
    g_click_completed: bool = False
    completed_primary_action: BalloonAction | None = None


class BalloonExecutor:
    """Execute visible decisions through the stochastic behavioral model."""

    def __init__(self, behavior: BehavioralModel, clock=None, initial_mouse_position=(0, 0)) -> None:
        self._behavior = behavior
        self._primary_scheduler = ActionScheduler(clock=clock)
        self._twin_scheduler = TwinActionScheduler(clock=clock)
        self._g_scheduler = ActionScheduler(clock=clock)
        self._current_mouse = initial_mouse_position
        self._active_primary: BalloonAction | TwinAction | None = None
        self._latest_g: BalloonAction | None = None
        self._g_visible = False
        self._idle_until = 0.0

    @property
    def primary_scheduler(self) -> ActionScheduler:
        return self._primary_scheduler

    @property
    def twin_scheduler(self) -> TwinActionScheduler:
        return self._twin_scheduler

    @property
    def g_scheduler(self) -> ActionScheduler:
        return self._g_scheduler

    def set_mouse_position(self, position: tuple[int, int]) -> None:
        self._current_mouse = position

    @property
    def latest_g(self) -> BalloonAction | None:
        return self._latest_g

    def tick(self, actions, g_action, input_backend, region_left, region_top, input_enabled) -> BalloonExecutionResult:
        now = self._clock()
        self._g_visible = g_action is not None
        self._latest_g = g_action

        clicks = 0
        input_timestamp = None
        action_timestamps = None
        twin_completion = None
        g_completed = False
        completed_primary_action = None

        # G is independently observable. If it disappears before dispatch,
        # cancel only the pending plan; an already-dispatched click may finish.
        if self._g_scheduler.active_action is not None and not self._g_scheduler.is_idle:
            active = self._g_scheduler.active_action
            g_position = self._to_screen(g_action.position, region_left, region_top) if g_action else None
            if active.phase.name in {"WAITING_REACTION", "MOVING"}:
                if g_position is None or (active.target_x, active.target_y) != (
                    round(self._behavioral_endpoint(active.behavioral_action)[0]),
                    round(self._behavioral_endpoint(active.behavioral_action)[1]),
                ):
                    self._g_scheduler.cancel()

        if self._g_visible and self._g_scheduler.is_idle and input_enabled:
            self._start_single(self._g_scheduler, self._latest_g, region_left, region_top)

        g_completion = self._g_scheduler.tick(input_backend) if self._g_scheduler.active_action else None
        if g_completion is not None:
            clicks += 1
            g_completed = True
            input_timestamp = g_completion.click_up

        if g_completion is not None and self._g_visible and input_enabled:
            self._start_single(self._g_scheduler, self._latest_g, region_left, region_top)

        if now >= self._idle_until and self._twin_scheduler.is_idle and self._primary_scheduler.is_idle:
            if actions and input_enabled:
                action = actions[0]
                if isinstance(action, TwinAction):
                    self._start_twin(action, region_left, region_top)
                else:
                    self._start_single(self._primary_scheduler, action, region_left, region_top)
                self._active_primary = action

        twin_completion = self._twin_scheduler.tick(input_backend)
        if twin_completion is not None:
            clicks += 2
            action_timestamps = twin_completion.first
            input_timestamp = max(twin_completion.first.click_up, twin_completion.second.click_up)
            self._current_mouse = (
                round(self._behavioral_endpoint(self._twin_scheduler.active_actions[1].behavioral_action)[0]),
                round(self._behavioral_endpoint(self._twin_scheduler.active_actions[1].behavioral_action)[1]),
            ) if self._twin_scheduler.active_actions else self._current_mouse
            self._active_primary = None
            self._idle_until = self._clock() + self._behavior.sample_idle_gap_s()

        primary_completion = self._primary_scheduler.tick(input_backend)
        if primary_completion is not None:
            clicks += 1
            action_timestamps = primary_completion
            input_timestamp = primary_completion.click_up
            if isinstance(self._active_primary, BalloonAction):
                completed_primary_action = self._active_primary
            self._active_primary = None
            self._primary_scheduler.clear()
            self._idle_until = self._clock() + self._behavior.sample_idle_gap_s()

        return BalloonExecutionResult(
            clicks_sent=clicks,
            input_timestamp=input_timestamp,
            action_timestamps=action_timestamps,
            twin_completion=twin_completion,
            g_click_completed=g_completed,
            completed_primary_action=completed_primary_action,
        )

    def _start_single(self, scheduler, action: BalloonAction | None, region_left: int, region_top: int) -> None:
        if action is None:
            return
        screen_x, screen_y = self._to_screen(action.position, region_left, region_top)
        behavioral_action = self._plan(action, screen_x, screen_y)
        scheduler.start_action(behavioral_action, screen_x, screen_y)

    def _start_twin(self, action: TwinAction, region_left: int, region_top: int) -> None:
        first, second = action.targets
        first_xy = self._to_screen(first.position, region_left, region_top)
        second_xy = self._to_screen(second.position, region_left, region_top)
        first_behavior = self._plan(first, *first_xy)
        second_behavior = self._plan(second, *second_xy)
        delay = self._behavior.sample_duo_delta_s()
        self._twin_scheduler.start_action(
            first_behavior, *first_xy, second_behavior, *second_xy,
            max_click_delta_ms=500.0, inter_click_delay_s=delay,
        )

    def _plan(self, action: BalloonAction, target_x: int, target_y: int) -> BehavioralAction:
        return self._behavior.plan_action(
            current_x=float(self._current_mouse[0]),
            current_y=float(self._current_mouse[1]),
            target_x=float(target_x),
            target_y=float(target_y),
            target_width=float(action.target_width),
            target_metadata=action.metadata,
            acquisition_timestamp=action.metadata.get("first_seen_timestamp") if action.metadata else None,
        )

    @staticmethod
    def _behavioral_endpoint(action: BehavioralAction) -> tuple[float, float]:
        return action.click_position

    @staticmethod
    def _to_screen(position, region_left, region_top):
        return frame_to_screen(position[0], position[1], region_left, region_top)
