from __future__ import annotations

from dataclasses import dataclass, replace
import time
from typing import Optional, Protocol

from behavior.action_scheduler import ActionScheduler, ActionTimestamps, TwinActionCompletion, TwinActionScheduler
from behavior.model import BehavioralAction, BehavioralModel
from game_profiles.balloon_types import BalloonAction, TwinAction
from input.coordinates import frame_to_screen
from telemetry.outcome import SessionOutcome
from telemetry.timing import now_monotonic
from diagnostics.downstream import DownstreamDiagnostic


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
    outcomes: tuple[SessionOutcome, ...] = ()


class BalloonExecutor:
    """Execute visible decisions through the stochastic behavioral model."""

    def __init__(self, behavior: BehavioralModel, clock=None, initial_mouse_position=(0, 0), diagnostic: DownstreamDiagnostic | None = None, fast_execution: bool = False) -> None:
        self._clock = clock or now_monotonic
        self._behavior = behavior
        self._diagnostic = diagnostic
        self._fast_execution = fast_execution
        self._primary_scheduler = ActionScheduler(clock=clock, diagnostic=diagnostic)
        self._twin_scheduler = TwinActionScheduler(clock=clock, diagnostic=diagnostic)
        self._g_scheduler = ActionScheduler(clock=clock, diagnostic=diagnostic)
        self._current_mouse = initial_mouse_position
        self._active_primary: BalloonAction | TwinAction | None = None
        self._latest_g: BalloonAction | None = None
        self._g_visible = False
        self._idle_until = 0.0
        self._g_decision_target_id = None
        self._g_should_execute = True
        self._last_behavior_plan_ms = 0.0

    @property
    def last_behavior_plan_ms(self) -> float:
        return self._last_behavior_plan_ms

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

    def tick(self, actions, g_action, input_backend, region_left, region_top, input_enabled, notifications=(), multicolor_specials=(), duration_s=None) -> BalloonExecutionResult:
        now = self._clock()
        self._last_behavior_plan_ms = 0.0
        self._g_visible = g_action is not None
        if self._diagnostic:
            self._diagnostic.log("executor_frame", now=now, actions=[self._action_info(a) for a in actions], g_action=self._action_info(g_action), notifications=[self._action_info(a) for a in notifications], multicolor_specials=[self._action_info(a) for a in multicolor_specials], input_enabled=input_enabled, active_primary=self._active_info(self._primary_scheduler.active_action), active_twin=self._active_twin_info(), active_g=self._active_info(self._g_scheduler.active_action), idle_until=self._idle_until)
        if g_action is None:
            self._g_decision_target_id = None
            self._g_should_execute = True
        elif g_action.target_id != self._g_decision_target_id:
            self._g_decision_target_id = g_action.target_id
            self._g_should_execute = not getattr(self._behavior, "should_abandon_balloon_g", lambda: False)()
        actions_allowed = getattr(self._behavior, "should_start_new_action", lambda _duration: True)(duration_s)
        self._latest_g = g_action

        if self._fast_execution:
            return self._tick_fast(
                actions,
                g_action,
                input_backend,
                region_left,
                region_top,
                input_enabled,
            )

        # A primary plan is created from a prior visual observation. Before the
        # scheduler reaches mouse-down, require that its target is still present
        # in the latest decision candidates. This is deliberately limited to
        # pre-dispatch phases; once mouse-down has occurred the click is already
        # in flight and is not cancelled from later visual changes.
        self._cancel_stale_primary(actions, notifications, multicolor_specials)

        if self._diagnostic:
            self._diagnostic.log("behavior_gate", gate="new_action", allowed=actions_allowed, **self._behavior_gate_state(duration_s))

        clicks = 0
        input_timestamp = None
        action_timestamps = None
        twin_completion = None
        g_completed = False
        completed_primary_action = None
        outcomes: list[SessionOutcome] = []

        # G is independently observable. If it disappears before dispatch,
        # cancel only the pending plan; an already-dispatched click may finish.
        if self._g_scheduler.active_action is not None and not self._g_scheduler.is_idle:
            active = self._g_scheduler.active_action
            # A disappearing G is observable; a moving/retargeted G is not
            # authoritative enough to cancel solely from endpoint noise.
            if active.phase.name in {"WAITING_REACTION", "MOVING"}:
                g_position = self._to_screen(g_action.position, region_left, region_top) if g_action else None
                if g_position is None or (
                    active.intended_target_x is not None
                    and active.intended_target_y is not None
                    and (active.intended_target_x, active.intended_target_y) != g_position
                ):
                    self._g_scheduler.cancel()

        if self._g_visible and self._diagnostic:
            self._diagnostic.log("action_start_attempt", lane="g", target=self._action_info(self._latest_g), accepted=self._g_should_execute and self._g_scheduler.is_idle and input_enabled and actions_allowed, reason=self._g_start_block_reason(input_enabled, actions_allowed))
        if self._g_visible and self._g_should_execute and self._g_scheduler.is_idle and input_enabled and actions_allowed:
            self._start_single(self._g_scheduler, self._latest_g, region_left, region_top)

        g_completion = self._g_scheduler.tick(input_backend) if self._g_scheduler.active_action else None
        if g_completion is not None:
            clicks += 1
            g_completed = True
            input_timestamp = g_completion.click_up
            g_active = self._g_scheduler.active_action
            if g_active is not None:
                outcomes.append(self._make_outcome(g_active.behavioral_action, g_completion, g_active.target_x, g_active.target_y, "g_symbol", target_id=self._latest_g.target_id if self._latest_g else None))

        if g_completion is not None and self._g_visible and input_enabled and actions_allowed:
            self._start_single(self._g_scheduler, self._latest_g, region_left, region_top)

        actions_allowed = getattr(self._behavior, "should_start_new_action", lambda _duration: True)(duration_s)
        if self._diagnostic:
            self._diagnostic.log("executor_gate", idle_gap_open=now >= self._idle_until, idle_until=self._idle_until, primary_idle=self._primary_scheduler.is_idle, twin_idle=self._twin_scheduler.is_idle, actions_allowed=actions_allowed, input_enabled=input_enabled)
        if self._diagnostic and (actions or notifications or multicolor_specials) and not (now >= self._idle_until and self._twin_scheduler.is_idle and self._primary_scheduler.is_idle and actions_allowed):
            reasons = []
            if now < self._idle_until:
                reasons.append("idle_gap")
            if not self._twin_scheduler.is_idle:
                reasons.append("twin_scheduler_busy")
            if not self._primary_scheduler.is_idle:
                reasons.append("primary_scheduler_busy")
            if not actions_allowed:
                reasons.append("behavioral_start_gate")
            if not input_enabled:
                reasons.append("input_disabled")
            self._diagnostic.log("action_start_attempt", lane="primary_or_twin", target=None, accepted=False, reason=reasons)
        if now >= self._idle_until and self._twin_scheduler.is_idle and self._primary_scheduler.is_idle and actions_allowed:
            if input_enabled:
                action = None
                gemini_gate_result = None
                notification_gate_result = None
                if notifications and multicolor_specials:
                    action = multicolor_specials[0]
                    gemini_gate_result = True
                    if self._diagnostic:
                        self._diagnostic.log("gemini_gate", candidate_count=len(multicolor_specials), notification_count=len(notifications), selected=True, reason="notification_present")
                elif notifications:
                    notification_gate_result = getattr(self._behavior, "should_click_notification", lambda: False)()
                    if notification_gate_result:
                        action = notifications[0]
                    if self._diagnostic:
                        self._diagnostic.log("notification_gate", candidate_count=len(notifications), selected=notification_gate_result, reason="behavioral_notification_gate")
                elif multicolor_specials:
                    gemini_gate_result = getattr(self._behavior, "should_select_gemini_without_notification", lambda: False)()
                    if gemini_gate_result:
                        action = multicolor_specials[0]
                    elif actions:
                        action = actions[0]
                    if self._diagnostic:
                        self._diagnostic.log("gemini_gate", candidate_count=len(multicolor_specials), notification_count=0, selected=gemini_gate_result, reason="behavioral_gemini_without_notification_probability")
                elif actions:
                    action = actions[0]
                # G precursors are a prerequisite game mechanic, not an
                # optional Super-G behavior sample. Never apply the legacy
                # should_fail_super_g() gate to them.
                if action is not None and self._diagnostic:
                    self._diagnostic.log("action_start_attempt", lane="primary_or_twin", target=self._action_info(action), accepted=True, reason=None)
                elif action is None and self._diagnostic:
                    self._diagnostic.log("action_start_attempt", lane="primary_or_twin", target=None, accepted=False, reason="no_selectable_action_or_behavioral_special_failure")
                if action is not None:
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
            if isinstance(self._active_primary, TwinAction):
                first_b, second_b = self._active_primary.targets
                if self._twin_scheduler.active_actions is not None:
                    first_a, second_a = self._twin_scheduler.active_actions
                    outcomes.append(self._make_outcome(first_a.behavioral_action, twin_completion.first, first_a.target_x, first_a.target_y, first_b.target_type.value, twin_completion.click_down_delta_ms, first_b.target_id))
                    outcomes.append(self._make_outcome(second_a.behavioral_action, twin_completion.second, second_a.target_x, second_a.target_y, second_b.target_type.value, twin_completion.click_down_delta_ms, second_b.target_id))
            self._active_primary = None
            idle_gap = self._behavior.sample_idle_gap_s() if hasattr(self._behavior, 'sample_idle_gap_s') else 0.0
            self._idle_until = self._clock() + idle_gap
            if self._diagnostic:
                self._diagnostic.log("action_complete", lane="twin", idle_gap_s=idle_gap, idle_until=self._idle_until)

        primary_completion = self._primary_scheduler.tick(input_backend)
        if primary_completion is not None:
            clicks += 1
            action_timestamps = primary_completion
            input_timestamp = primary_completion.click_up
            if isinstance(self._active_primary, BalloonAction):
                completed_primary_action = self._active_primary
                active = self._primary_scheduler.active_action
                if active is not None:
                    outcomes.append(self._make_outcome(active.behavioral_action, primary_completion, active.target_x, active.target_y, completed_primary_action.target_type.value, target_id=completed_primary_action.target_id))
            self._active_primary = None
            self._primary_scheduler.clear()
            idle_gap = self._behavior.sample_idle_gap_s() if hasattr(self._behavior, 'sample_idle_gap_s') else 0.0
            # Give the four-node G precursor sequence a deterministic minimum
            # inter-click gap. The normal behavioral model still contributes
            # its sampled idle gap when that is longer.
            precursor_gap_s = 0.100 if (
                completed_primary_action is not None
                and completed_primary_action.metadata.get("g_precursor", False)
            ) else 0.0
            self._idle_until = self._clock() + max(idle_gap, precursor_gap_s)
            if self._diagnostic:
                self._diagnostic.log(
                    "action_complete",
                    lane="primary",
                    target=self._action_info(completed_primary_action),
                    idle_gap_s=idle_gap,
                    precursor_gap_s=precursor_gap_s,
                    idle_until=self._idle_until,
                )

        return BalloonExecutionResult(
            clicks_sent=clicks,
            input_timestamp=input_timestamp,
            action_timestamps=action_timestamps,
            twin_completion=twin_completion,
            g_click_completed=g_completed,
            completed_primary_action=completed_primary_action,
            outcomes=tuple(outcomes),
        )

    def _tick_fast(
        self,
        actions,
        g_action,
        input_backend,
        region_left: int,
        region_top: int,
        input_enabled: bool,
    ) -> BalloonExecutionResult:
        """Dispatch visible balloon decisions immediately through the input backend.

        This explicit fast path bypasses behavioral reaction, movement, click-hold,
        and idle-gap scheduling. The normal stochastic scheduler remains untouched
        and is used whenever fast_execution is False.
        """
        if not input_enabled:
            return BalloonExecutionResult()

        clicks = 0
        input_timestamp = None
        action_timestamps = None
        twin_completion = None
        g_completed = False
        completed_primary_action = None
        outcomes: list[SessionOutcome] = []

        if g_action is not None:
            g_screen = self._to_screen(g_action.position, region_left, region_top)
            g_behavior = self._fast_behavioral_action(g_screen, g_action.target_width)
            g_ts = self._fast_click(input_backend, g_screen)
            clicks += 1
            g_completed = True
            input_timestamp = g_ts.click_up
            action_timestamps = g_ts
            outcomes.append(
                self._make_outcome(
                    g_behavior,
                    g_ts,
                    g_screen[0],
                    g_screen[1],
                    "g_symbol",
                    target_id=g_action.target_id,
                    execution_status="fast",
                )
            )
            if self._diagnostic:
                self._diagnostic.log(
                    "fast_click",
                    lane="g",
                    target=self._action_info(g_action),
                    click_down_ts=g_ts.click_down,
                    click_up_ts=g_ts.click_up,
                )

        if not actions:
            return BalloonExecutionResult(
                clicks_sent=clicks,
                input_timestamp=input_timestamp,
            )

        action = actions[0]
        if isinstance(action, TwinAction):
            first, second = action.targets
            first_xy = self._to_screen(first.position, region_left, region_top)
            second_xy = self._to_screen(second.position, region_left, region_top)
            first_behavior = self._fast_behavioral_action(first_xy, first.target_width)
            second_behavior = self._fast_behavioral_action(second_xy, second.target_width)
            first_down, first_up, second_down, second_up = input_backend.click_pair(
                first_xy,
                second_xy,
                0.0,
            )
            first_ts = ActionTimestamps(
                action_planned=first_down,
                reaction_start=first_down,
                reaction_end=first_down,
                movement_start=first_down,
                movement_end=first_down,
                click_down=first_down,
                click_up=first_up,
                action_complete=first_up,
            )
            second_ts = ActionTimestamps(
                action_planned=second_down,
                reaction_start=second_down,
                reaction_end=second_down,
                movement_start=second_down,
                movement_end=second_down,
                click_down=second_down,
                click_up=second_up,
                action_complete=second_up,
            )
            twin_completion = TwinActionCompletion(
                first_ts,
                second_ts,
                abs(first_down - second_down) * 1000.0,
                0.0,
            )
            clicks += 2
            input_timestamp = max(first_up, second_up)
            action_timestamps = first_ts
            outcomes.extend(
                (
                    self._make_outcome(
                        first_behavior,
                        first_ts,
                        first_xy[0],
                        first_xy[1],
                        first.target_type.value,
                        target_id=first.target_id,
                        execution_status="fast",
                    ),
                    self._make_outcome(
                        second_behavior,
                        second_ts,
                        second_xy[0],
                        second_xy[1],
                        second.target_type.value,
                        target_id=second.target_id,
                        execution_status="fast",
                    ),
                )
            )
            completed_primary_action = action
        else:
            screen_xy = self._to_screen(action.position, region_left, region_top)
            behavioral_action = self._fast_behavioral_action(
                screen_xy,
                action.target_width,
            )
            action_timestamps = self._fast_click(input_backend, screen_xy)
            clicks += 1
            input_timestamp = action_timestamps.click_up
            completed_primary_action = action
            outcomes.append(
                self._make_outcome(
                    behavioral_action,
                    action_timestamps,
                    screen_xy[0],
                    screen_xy[1],
                    action.target_type.value,
                    target_id=action.target_id,
                    execution_status="fast",
                )
            )
            if self._diagnostic:
                self._diagnostic.log(
                    "fast_click",
                    lane="primary",
                    target=self._action_info(action),
                    click_down_ts=action_timestamps.click_down,
                    click_up_ts=action_timestamps.click_up,
                )

        return BalloonExecutionResult(
            clicks_sent=clicks,
            input_timestamp=input_timestamp,
            action_timestamps=action_timestamps,
            twin_completion=twin_completion,
            g_click_completed=g_completed,
            completed_primary_action=completed_primary_action,
            outcomes=tuple(outcomes),
        )

    def _fast_click(self, input_backend, screen_xy: tuple[int, int]) -> ActionTimestamps:
        action_planned = self._clock()
        input_backend.move_to(*screen_xy)
        movement_end = self._clock()
        click_down = input_backend.left_down()
        click_up = input_backend.left_up()
        return ActionTimestamps(
            action_planned=action_planned,
            reaction_start=action_planned,
            reaction_end=movement_end,
            movement_start=action_planned,
            movement_end=movement_end,
            click_down=click_down,
            click_up=click_up,
            action_complete=click_up,
        )

    @staticmethod
    def _fast_behavioral_action(
        screen_xy: tuple[int, int],
        target_width: float,
    ) -> BehavioralAction:
        now = time.perf_counter()
        return BehavioralAction(
            reaction_time_s=0.0,
            movement_time_s=0.0,
            click_duration_s=0.0,
            trajectory=[],
            total_time_s=0.0,
            start_position=(float(screen_xy[0]), float(screen_xy[1])),
            target_position=(float(screen_xy[0]), float(screen_xy[1])),
            click_position=(float(screen_xy[0]), float(screen_xy[1])),
            distance=0.0,
            target_width=max(1.0, float(target_width)),
            fitts_id=0.0,
            endpoint_error_px=0.0,
            pause_duration_s=0.0,
            is_miss=False,
            is_delayed=False,
            session_modifier=0.0,
            warmup_factor=0.0,
            fatigue_factor=0.0,
            planned_at=now,
            acquisition_timestamp=None,
            scheduler_delay_s=0.0,
        )

    def _cancel_stale_primary(self, actions, notifications, multicolor_specials) -> None:
        if self._primary_scheduler.is_idle and self._twin_scheduler.is_idle:
            return

        active = self._primary_scheduler.active_action
        if active is not None and active.phase.name in {"WAITING_REACTION", "MOVING", "CLICK_DOWN"}:
            target_id = self._active_primary.target_id if isinstance(self._active_primary, BalloonAction) else None
            if target_id is not None and not self._target_id_observed(
                target_id, actions, notifications, multicolor_specials
            ):
                self._primary_scheduler.cancel()
                if self._diagnostic:
                    self._diagnostic.log(
                        "stale_action_cancel",
                        lane="primary",
                        target_id=target_id,
                        phase=active.phase.name,
                        reason="target_absent_from_latest_decision_observation",
                    )

        active_twin = self._twin_scheduler.active_actions
        if active_twin is not None and not self._twin_scheduler.is_idle:
            target_ids = {
                target.target_id
                for target in self._active_primary.targets
            } if isinstance(self._active_primary, TwinAction) else set()
            if target_ids and not target_ids.issubset(
                self._observed_target_ids(actions, notifications, multicolor_specials)
            ):
                self._twin_scheduler.clear()
                if self._diagnostic:
                    self._diagnostic.log(
                        "stale_action_cancel",
                        lane="twin",
                        target_ids=sorted(target_ids),
                        reason="target_absent_from_latest_decision_observation",
                    )

    @classmethod
    def _observed_target_ids(cls, actions, notifications, multicolor_specials) -> set[int]:
        ids: set[int] = set()
        for candidate in tuple(actions) + tuple(notifications) + tuple(multicolor_specials):
            if isinstance(candidate, TwinAction):
                for target in candidate.targets:
                    if target.target_id is not None:
                        ids.add(target.target_id)
            elif candidate.target_id is not None:
                ids.add(candidate.target_id)
        return ids

    @classmethod
    def _target_id_observed(cls, target_id, actions, notifications, multicolor_specials) -> bool:
        return target_id in cls._observed_target_ids(actions, notifications, multicolor_specials)

    def shutdown(self, input_backend, reason: str = "session_end") -> tuple[SessionOutcome, ...]:
        """Stop schedulers and release any button held by an interrupted action."""
        outcomes: list[SessionOutcome] = []

        active = self._primary_scheduler.active_action
        if active is not None and not self._primary_scheduler.is_idle:
            action = self._active_primary
            timestamps = self._primary_scheduler.cancel(input_backend=input_backend, reason=reason)
            if timestamps is not None and isinstance(action, BalloonAction):
                outcomes.append(
                    self._make_outcome(
                        active.behavioral_action,
                        timestamps,
                        active.target_x,
                        active.target_y,
                        action.target_type.value,
                        target_id=action.target_id,
                        execution_status="interrupted",
                    )
                )
            self._active_primary = None

        twin_active = self._twin_scheduler.active_actions
        if twin_active is not None and not self._twin_scheduler.is_idle:
            action = self._active_primary
            self._twin_scheduler.clear()
            if isinstance(action, TwinAction):
                for active_action, target in zip(twin_active, action.targets):
                    timestamps = active_action.timestamps
                    timestamps.interrupted = True
                    timestamps.cancel_reason = reason
                    timestamps.action_complete = self._clock()
                    outcomes.append(
                        self._make_outcome(
                            active_action.behavioral_action,
                            timestamps,
                            active_action.target_x,
                            active_action.target_y,
                            target.target_type.value,
                            target_id=target.target_id,
                            execution_status="interrupted",
                        )
                    )
            self._active_primary = None

        active_g = self._g_scheduler.active_action
        if active_g is not None and not self._g_scheduler.is_idle:
            timestamps = self._g_scheduler.cancel(input_backend=input_backend, reason=reason)
            if timestamps is not None:
                outcomes.append(
                    self._make_outcome(
                        active_g.behavioral_action,
                        timestamps,
                        active_g.target_x,
                        active_g.target_y,
                        "g_symbol",
                        target_id=self._latest_g.target_id if self._latest_g else None,
                        execution_status="interrupted",
                    )
                )

        if self._diagnostic:
            self._diagnostic.log("session_shutdown", reason=reason, interrupted_actions=len(outcomes))
        return tuple(outcomes)

    def _start_single(self, scheduler, action: BalloonAction | None, region_left: int, region_top: int) -> None:
        if action is None:
            return
        screen_x, screen_y = self._to_screen(action.position, region_left, region_top)
        behavioral_action = self._plan(action, screen_x, screen_y)
        if not isinstance(self._behavior, BehavioralModel):
            behavioral_action = replace(
                behavioral_action,
                click_position=(float(screen_x), float(screen_y)),
                target_position=(float(screen_x), float(screen_y)),
            )
        scheduler.start_action(behavioral_action, screen_x, screen_y)

    def _start_twin(self, action: TwinAction, region_left: int, region_top: int) -> None:
        first, second = action.targets
        first_xy = self._to_screen(first.position, region_left, region_top)
        second_xy = self._to_screen(second.position, region_left, region_top)
        first_behavior = self._plan(first, *first_xy)
        second_behavior = self._plan(second, *second_xy)
        if not isinstance(self._behavior, BehavioralModel):
            first_behavior = replace(first_behavior, click_position=(float(first_xy[0]), float(first_xy[1])), target_position=(float(first_xy[0]), float(first_xy[1])))
            second_behavior = replace(second_behavior, click_position=(float(second_xy[0]), float(second_xy[1])), target_position=(float(second_xy[0]), float(second_xy[1])))
        delay = self._behavior.sample_duo_delta_s() if hasattr(self._behavior, "sample_duo_delta_s") else 0.0
        max_delta = 500.0 if isinstance(self._behavior, BehavioralModel) else 30.0
        self._twin_scheduler.start_action(
            first_behavior, *first_xy, second_behavior, *second_xy,
            max_click_delta_ms=max_delta, inter_click_delay_s=delay,
        )

    def _plan(self, action: BalloonAction, target_x: int, target_y: int) -> BehavioralAction:
        metadata = dict(action.metadata or {})
        metadata["is_twin"] = action.is_twin
        t0 = time.perf_counter_ns()
        result = self._behavior.plan_action(
            current_x=float(self._current_mouse[0]),
            current_y=float(self._current_mouse[1]),
            target_x=float(target_x),
            target_y=float(target_y),
            target_width=float(action.target_width),
            target_metadata=metadata,
            acquisition_timestamp=metadata.get("first_seen_timestamp"),
        )
        self._last_behavior_plan_ms += (time.perf_counter_ns() - t0) / 1e6
        return result

    @staticmethod
    def _behavioral_endpoint(action: BehavioralAction) -> tuple[float, float]:
        return action.click_position

    @staticmethod
    def _make_outcome(
        behavioral_action: BehavioralAction,
        timestamps: ActionTimestamps,
        screen_x: int,
        screen_y: int,
        target_type: str,
        duo_delta_ms: float | None = None,
        target_id: int | None = None,
        execution_status: str = "completed",
    ) -> SessionOutcome:
        observed_reaction = None
        estimated_click_tick = None
        timing_error_ms = None
        if behavioral_action.acquisition_timestamp is not None and timestamps.click_down:
            observed_reaction = max(
                0.0,
                (timestamps.click_down - behavioral_action.acquisition_timestamp) / (1000.0 / 60.0),
            )
            # This is a relative tick estimate from first-seen, not an engine tick.
            estimated_click_tick = observed_reaction
            timing_error_ms = (observed_reaction - behavioral_action.intended_reaction_ticks) * (1000.0 / 60.0)
        scheduler_delay_ms = max(0.0, (timestamps.reaction_end - timestamps.action_planned) * 1000.0 - behavioral_action.reaction_time_s * 1000.0)
        movement_duration_ms = max(0.0, (timestamps.movement_end - timestamps.movement_start) * 1000.0)
        is_gemini = target_type == "multicolor_special"
        return SessionOutcome(
            target_id=target_id,
            target_type=target_type,
            target_first_seen_ts=behavioral_action.acquisition_timestamp,
            acquisition_ts=behavioral_action.acquisition_timestamp,
            planned_dispatch_ts=timestamps.action_planned or behavioral_action.planned_at,
            behavioral_planning_ts=behavioral_action.planned_at,
            dispatch_ts=timestamps.click_down or timestamps.click_up,
            actual_click_down_ts=timestamps.click_down or None,
            estimated_reaction_ticks=observed_reaction,
            estimated_acquisition_click_ticks=observed_reaction,
            estimated_click_tick=estimated_click_tick,
            click_tick_modulo=(int(round(estimated_click_tick)) % 3) if estimated_click_tick is not None else None,
            timing_error_ms=timing_error_ms,
            intended_vs_observed_timing_error_ms=timing_error_ms,
            screen_click_x=int(screen_x),
            screen_click_y=int(screen_y),
            endpoint_error_px=behavioral_action.endpoint_error_px,
            target_radius_px=behavioral_action.target_width / 2.0,
            intended_reaction_ticks=behavioral_action.intended_reaction_ticks,
            intended_reaction_s=behavioral_action.reaction_time_s,
            modeled_perception_ticks=behavioral_action.perception_ticks,
            scheduler_delay_ms=scheduler_delay_ms,
            movement_duration_ms=movement_duration_ms,
            duo_delta_ms=duo_delta_ms,
            gemini_observed=is_gemini,
            gemini_selected=is_gemini,
            gemini_executed=(timestamps.click_down > 0),
            gemini_completed=None,
            execution_status=execution_status,
            observation_notes=["UNOBSERVABLE THROUGH CURRENT INTERFACE: authoritative spawnTick/popTick/hit state; click tick is a relative estimate from first-seen"],
        )

    def _behavior_gate_state(self, duration_s):
        if not isinstance(self._behavior, BehavioralModel):
            return {}
        elapsed = self._behavior.session_elapsed_s()
        cutoff = self._behavior.activity_cutoff_s()
        deadline = None if not duration_s or duration_s <= 0 else duration_s - cutoff
        return {
            "session_started": self._behavior._session_started,
            "session_elapsed_s": elapsed,
            "start_delay_s": self._behavior.start_delay_s,
            "actions_ready": self._behavior.actions_ready(),
            "activity_cutoff_s": cutoff,
            "final_activity_deadline_s": deadline,
            "final_activity_cutoff_active": bool(deadline is not None and elapsed >= deadline),
        }

    def _g_start_block_reason(self, input_enabled, actions_allowed):
        if not self._g_should_execute:
            return "behavioral_g_abandon"
        if not self._g_scheduler.is_idle:
            return "g_scheduler_busy"
        if not input_enabled:
            return "input_disabled"
        if not actions_allowed:
            return "behavioral_start_gate"
        return None

    @staticmethod
    def _action_info(action):
        if action is None:
            return None
        if isinstance(action, TwinAction):
            return {"type": "TwinAction", "targets": [BalloonExecutor._action_info(t) for t in action.targets]}
        return {
            "type": action.target_type.value,
            "id": action.target_id,
            "color": action.color.value if action.color else None,
            "position": list(action.position),
            "width": action.target_width,
        }

    def _active_info(self, active):
        if active is None:
            return None
        return {
            "phase": active.phase.name,
            "target_x": active.target_x,
            "target_y": active.target_y,
            "intended_target_x": active.intended_target_x,
            "intended_target_y": active.intended_target_y,
        }

    def _active_twin_info(self):
        active = self._twin_scheduler.active_actions
        if active is None:
            return None
        return [{"phase": a.phase.name, "target_x": a.target_x, "target_y": a.target_y} for a in active]

    @staticmethod
    def _to_screen(position, region_left, region_top):
        return frame_to_screen(position[0], position[1], region_left, region_top)
