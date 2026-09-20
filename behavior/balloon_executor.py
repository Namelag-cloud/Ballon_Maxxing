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
        self._policy = execution_policy or FastExecutionPolicy(
            clock=self._clock,
            diagnostic=diagnostic,
        )
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

    def shutdown(self, input_backend, reason: str) -> tuple[SessionOutcome, ...]:
        """Finalize the active execution policy at session end."""
        return self._policy.shutdown(input_backend, reason)

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



