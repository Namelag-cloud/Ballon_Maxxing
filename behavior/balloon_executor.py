from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol

from behavior.action_scheduler import (
    ActionScheduler,
    ActionTimestamps,
    TwinActionCompletion,
    TwinActionScheduler,
)
from behavior.model import BehavioralAction, BehavioralModel
from game_profiles.balloon_types import BalloonAction, TwinAction
from input.coordinates import frame_to_screen


class PairInputBackend(Protocol):
    def click_pair(
        self,
        first: tuple[int, int],
        second: tuple[int, int],
    ) -> tuple[float, float, float, float]:
        """Dispatch two complete clicks and return down/up timestamps."""


@dataclass(frozen=True)
class BalloonExecutionResult:
    clicks_sent: int = 0
    input_timestamp: Optional[float] = None
    action_timestamps: Optional[ActionTimestamps] = None
    twin_completion: Optional[TwinActionCompletion] = None
    g_click_completed: bool = False
    completed_primary_action: BalloonAction | None = None


class BalloonExecutor:
    """Execute the decision-layer balloon contracts.

    The primary stream has one active action at a time. Twins use the
    coordinated scheduler, while G has an independent scheduler. G visibility
    is supplied every frame; disappearance prevents another G action from
    being scheduled but never uses a click counter as its lifetime condition.
    """

    def __init__(
        self,
        behavior: BehavioralModel,
        clock=None,
        initial_mouse_position: tuple[int, int] = (0, 0),
    ) -> None:
        self._behavior = behavior
        self._primary_scheduler = ActionScheduler(clock=clock)
        self._twin_scheduler = TwinActionScheduler(clock=clock)
        self._g_scheduler = ActionScheduler(clock=clock)
        self._current_mouse = initial_mouse_position
        self._active_primary: BalloonAction | TwinAction | None = None
        self._latest_g: BalloonAction | None = None
        self._g_visible = False

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

    def tick(
        self,
        actions: tuple[BalloonAction | TwinAction, ...] | list[BalloonAction | TwinAction],
        g_action: BalloonAction | None,
        input_backend: PairInputBackend,
        region_left: int,
        region_top: int,
        input_enabled: bool,
    ) -> BalloonExecutionResult:
        self._g_visible = g_action is not None
        self._latest_g = g_action

        clicks = 0
        input_timestamp: Optional[float] = None
        action_timestamps: Optional[ActionTimestamps] = None
        twin_completion: Optional[TwinActionCompletion] = None
        g_completed = False
        completed_primary_action: BalloonAction | None = None

        # Refresh an in-flight G plan before its click phase when vision has
        # moved the symbol. If G disappears before a click is dispatched, cancel
        # the pending plan; an already-dispatched click is allowed to finish.
        if self._g_scheduler.active_action is not None and not self._g_scheduler.is_idle:
            g_active = self._g_scheduler.active_action
            g_position = (
                *self._to_screen(self._latest_g.position, region_left, region_top),
            ) if self._latest_g is not None else None
            if g_active.phase.name in {"WAITING_REACTION", "MOVING"}:
                if g_position is None or (g_active.target_x, g_active.target_y) != g_position:
                    self._g_scheduler.cancel()

        # G is a concurrent lane. Start a visible G action before ticking the
        # scheduler so a newly observed symbol participates in this polling
        # cycle instead of waiting an extra cycle before reaction/movement.
        # A completed click may schedule the next click below, but that new
        # action intentionally waits for the next tick; only one G click is
        # dispatched per executor tick.
        if self._g_visible and self._g_scheduler.is_idle and input_enabled:
            self._start_single(
                self._g_scheduler,
                self._latest_g,
                input_backend,
                region_left,
                region_top,
            )

        g_completion = self._g_scheduler.tick(input_backend)
        if g_completion is not None:
            clicks += 1
            g_completed = True
            input_timestamp = g_completion.click_up
            if self._g_scheduler.active_action is not None:
                self._current_mouse = (
                    self._g_scheduler.active_action.target_x,
                    self._g_scheduler.active_action.target_y,
                )

        if g_completion is not None and self._g_visible and input_enabled:
            self._start_single(
                self._g_scheduler,
                self._latest_g,
                input_backend,
                region_left,
                region_top,
            )

        # If G disappears, refresh/cancellation above prevents any new action
        # from being scheduled. An already-dispatched click is allowed to
        # finish through the scheduler. No counter is involved in G lifetime.

        if self._twin_scheduler.is_idle and self._primary_scheduler.is_idle:
            if actions and input_enabled:
                action = actions[0]
                if isinstance(action, TwinAction):
                    self._start_twin(
                        action,
                        input_backend,
                        region_left,
                        region_top,
                    )
                else:
                    self._start_single(
                        self._primary_scheduler,
                        action,
                        input_backend,
                        region_left,
                        region_top,
                    )
                self._active_primary = action

        twin_completion = self._twin_scheduler.tick(input_backend)
        if twin_completion is not None:
            clicks += 2
            action_timestamps = twin_completion.first
            self._current_mouse = (
                self._twin_scheduler.active_actions[1].target_x,
                self._twin_scheduler.active_actions[1].target_y,
            ) if self._twin_scheduler.active_actions is not None else self._current_mouse
            input_timestamp = max(
                twin_completion.first.click_up,
                twin_completion.second.click_up,
            )
            self._active_primary = None

        primary_completion = self._primary_scheduler.tick(input_backend)
        if primary_completion is not None:
            clicks += 1
            action_timestamps = primary_completion
            input_timestamp = primary_completion.click_up
            if self._primary_scheduler.active_action is not None:
                self._current_mouse = (
                    self._primary_scheduler.active_action.target_x,
                    self._primary_scheduler.active_action.target_y,
                )
            if isinstance(self._active_primary, BalloonAction):
                completed_primary_action = self._active_primary
            self._active_primary = None
            self._primary_scheduler.clear()

        return BalloonExecutionResult(
            clicks_sent=clicks,
            input_timestamp=input_timestamp,
            action_timestamps=action_timestamps,
            twin_completion=twin_completion,
            g_click_completed=g_completed,
            completed_primary_action=completed_primary_action,
        )

    def _start_single(
        self,
        scheduler: ActionScheduler,
        action: BalloonAction | None,
        input_backend: PairInputBackend,
        region_left: int,
        region_top: int,
    ) -> None:
        if action is None:
            return
        screen_x, screen_y = self._to_screen(
            action.position, region_left, region_top
        )
        behavioral_action = self._plan(action, screen_x, screen_y)
        scheduler.start_action(behavioral_action, screen_x, screen_y)

    def _start_twin(
        self,
        action: TwinAction,
        input_backend: PairInputBackend,
        region_left: int,
        region_top: int,
    ) -> None:
        first, second = action.targets
        first_xy = self._to_screen(first.position, region_left, region_top)
        second_xy = self._to_screen(second.position, region_left, region_top)

        first_behavior = self._plan(first, *first_xy)
        second_behavior = self._plan(second, *second_xy)

        self._twin_scheduler.start_action(
            first_behavior,
            first_xy[0],
            first_xy[1],
            second_behavior,
            second_xy[0],
            second_xy[1],
            max_click_delta_ms=action.max_click_delta_ms,
        )

    def _plan(
        self,
        action: BalloonAction,
        target_x: int,
        target_y: int,
    ) -> BehavioralAction:
        behavioral_action = self._behavior.plan_action(
            current_x=float(self._current_mouse[0]),
            current_y=float(self._current_mouse[1]),
            target_x=float(target_x),
            target_y=float(target_y),
            target_width=float(action.target_width),
            target_metadata=action.metadata,
        )

        if behavioral_action.is_miss:
            target_x += int(self._behavior.rng.normal(0, 50))
            target_y += int(self._behavior.rng.normal(0, 50))

        return behavioral_action

    @staticmethod
    def _to_screen(
        position: tuple[int, int],
        region_left: int,
        region_top: int,
    ) -> tuple[int, int]:
        return frame_to_screen(
            position[0],
            position[1],
            region_left,
            region_top,
        )
