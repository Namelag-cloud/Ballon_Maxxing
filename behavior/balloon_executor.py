from __future__ import annotations

from dataclasses import replace
import time

from behavior.action_scheduler import ActionScheduler, ActionTimestamps, TwinActionCompletion, TwinActionScheduler
from behavior.execution_policy import (
    BalloonExecutionResult,
    ExecutionContext,
    ExecutionPolicy,
)
from behavior.fast_policy import FastExecutionPolicy
from behavior.model import BehavioralAction, BehavioralModel
from decision.mechanics import DecisionResult as BalloonFrameDecision
from game_profiles.balloon_types import (
    BalloonAction,
    GameMode,
    SpecialSequenceStep,
    TwinAction,
)
from input.coordinates import frame_to_screen
from telemetry.outcome import SessionOutcome
from telemetry.timing import now_monotonic
from diagnostics.downstream import DownstreamDiagnostic


class BalloonExecutor:
    """Execute visible decisions through the stochastic behavioral model."""

    def __init__(
        self,
        behavior: BehavioralModel,
        clock=None,
        initial_mouse_position=(0, 0),
        diagnostic: DownstreamDiagnostic | None = None,
        fast_execution: bool = False,
        execution_policy: ExecutionPolicy | None = None,
    ) -> None:
        self._clock = clock or now_monotonic
        self._behavior = behavior
        self._diagnostic = diagnostic
        if execution_policy is not None:
            self._policy = execution_policy
        elif fast_execution:
            # Backward-compatible constructor path for existing tests/callers.
            # Production wiring constructs the policy in main.py.
            self._policy = FastExecutionPolicy(
                clock=self._clock,
                diagnostic=diagnostic,
            )
        else:
            self._policy = LegacyBehavioralExecutionPolicy(self)
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
    def current_mouse_position(self) -> tuple[int, int]:
        return self._current_mouse

    @property
    def latest_g(self) -> BalloonAction | None:
        return self._latest_g

    def execute(
        self,
        decision: BalloonFrameDecision,
        context: ExecutionContext,
        input_backend,
    ) -> BalloonExecutionResult:
        return self._policy.tick(decision, context, input_backend)

    def tick(
        self,
        actions,
        g_action,
        input_backend,
        region_left,
        region_top,
        input_enabled,
        notifications=(),
        multicolor_specials=(),
        duration_s=None,
    ) -> BalloonExecutionResult:
        decision = BalloonFrameDecision(
            actions=tuple(actions),
            g_action=g_action,
            multicolor_specials=tuple(multicolor_specials),
            notifications=tuple(notifications),
            mode=GameMode.NORMAL_PLAY,
            special_step=SpecialSequenceStep.COMPLETE,
        )
        context = ExecutionContext(
            frame_id=0,
            now=self._clock(),
            capture_timestamp=None,
            region_left=region_left,
            region_top=region_top,
            input_enabled=input_enabled,
            duration_s=duration_s,
            current_mouse_position=self._current_mouse,
        )
        return self.execute(decision, context, input_backend)

    def _tick_legacy(
        self,
        actions,
        g_action,
        input_backend,
        region_left,
        region_top,
        input_enabled,
        notifications=(),
        multicolor_specials=(),
        duration_s=None,
    ) -> BalloonExecutionResult:
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
                elif actions:
                    # Gameplay actions must not be starved by notifications.
                    # This is especially important for the mandatory G
                    # precursor sequence and for twin balloons.
                    action = actions[0]
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
                    if self._diagnostic:
                        self._diagnostic.log("gemini_gate", candidate_count=len(multicolor_specials), notification_count=0, selected=gemini_gate_result, reason="behavioral_gemini_without_notification_probability")
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

    def _cancel_stale_primary(self, actions, notifications, multicolor_specials) -> None:
        if self._primary_scheduler.is_idle and self._twin_scheduler.is_idle:
            return

        active = self._primary_scheduler.active_action
        if active is not None and active.phase.name in {"WAITING_REACTION", "MOVING", "CLICK_DOWN"}:
            target_id = self._active_primary.target_id if isinstance(self._active_primary, BalloonAction) else None
            if target_id is not None and not self._target_id_observed(
                target_id, actions, notifications, multicolor_specials
            ):
                # Tracker IDs can churn between frames even when the same
                # physical G precursor is still visible. Preserve stale-target
                # protection, but allow a precursor to follow a nearby
                # same-color replacement candidate.
                precursor_replacement = (
                    isinstance(self._active_primary, BalloonAction)
                    and self._active_primary.metadata.get("g_precursor", False)
                    and self._special_precursor_replacement_observed(
                        self._active_primary,
                        actions,
                    )
                )
                if not precursor_replacement:
                    self._primary_scheduler.cancel()
                    if self._diagnostic:
                        self._diagnostic.log(
                            "stale_action_cancel",
                            lane="primary",
                            target_id=target_id,
                            phase=active.phase.name,
                            reason="target_absent_from_latest_decision_observation",
                        )
                elif self._diagnostic:
                    self._diagnostic.log(
                        "stale_action_rebind",
                        lane="primary",
                        target_id=target_id,
                        phase=active.phase.name,
                        reason="g_precursor_same_color_nearby_replacement",
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

    @staticmethod
    def _special_precursor_replacement_observed(active: BalloonAction, actions) -> bool:
        if active.target_type.value != "special_sequence_balloon" or active.color is None:
            return False
        ax, ay = active.position
        # Precursor rings are relatively large; 48 px is deliberately wider
        # than normal frame-to-frame tracker jitter but small enough to avoid
        # rebinding to a different nearby target.
        max_distance_sq = 48 * 48
        for candidate in actions:
            if not isinstance(candidate, BalloonAction):
                continue
            if candidate.target_type.value != "special_sequence_balloon":
                continue
            if candidate.color != active.color:
                continue
            if candidate.target_id == active.target_id:
                return True
            dx = candidate.position[0] - ax
            dy = candidate.position[1] - ay
            if dx * dx + dy * dy <= max_distance_sq:
                return True
        return False

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
        return self._policy.shutdown(input_backend, reason)

    def _shutdown_legacy(self, input_backend, reason: str = "session_end") -> tuple[SessionOutcome, ...]:
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



class LegacyBehavioralExecutionPolicy:
    """Compatibility adapter for callers that still exercise the old scheduler."""

    def __init__(self, executor: BalloonExecutor) -> None:
        self._executor = executor

    def tick(
        self,
        decision: BalloonFrameDecision,
        context: ExecutionContext,
        input_backend,
    ) -> BalloonExecutionResult:
        return self._executor._tick_legacy(
            decision.actions,
            decision.g_action,
            input_backend,
            context.region_left,
            context.region_top,
            context.input_enabled,
            decision.notifications,
            decision.multicolor_specials,
            context.duration_s,
        )

    def shutdown(self, input_backend, reason: str) -> tuple[SessionOutcome, ...]:
        return self._executor._shutdown_legacy(input_backend, reason)
