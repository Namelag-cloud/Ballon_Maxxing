from __future__ import annotations

import time

from behavior.action_scheduler import ActionTimestamps, TwinActionCompletion
from behavior.model import BehavioralAction
from diagnostics.downstream import DownstreamDiagnostic
from game_profiles.balloon_types import BalloonAction, TwinAction
from input.coordinates import frame_to_screen
from telemetry.outcome import SessionOutcome
from telemetry.timing import now_monotonic

# Inter-click timing controls for already-planned action sequences.
# Tune these independently when experimenting with different cadence profiles.
NORMAL_CLICK_GAP_RANGE_S = (0.080, 0.110)
G_PRECURSOR_CLICK_GAP_RANGE_S = (0.080, 0.140)
SUPER_G_CLICK_GAP_RANGE_S = (0.100, 0.220)
MULTICOLOR_SPECIAL_CLICK_GAP_RANGE_S = (0.080, 0.140)
TWIN_CLICK_GAP_RANGE_S = (0.080, 0.150)
NORMAL_CLICK_OFFSET_PX_RANGE = (-2, 2)

from .execution_policy import (
    BalloonExecutionResult,
    ExecutionContext,
    ExecutionPolicy,
    InputBackend,
)


class FastExecutionPolicy:
    """Mechanical extraction of BalloonExecutor's immediate-dispatch fast path."""

    def __init__(
        self,
        *,
        clock=now_monotonic,
        diagnostic: DownstreamDiagnostic | None = None,
    ) -> None:
        self._clock = clock
        self._diagnostic = diagnostic
        self._rng = __import__("random").Random()
        self._last_click_down_ts: float | None = None
        self._recent_fast_clicks: list[
            tuple[float, int | None, tuple[int, int], float]
        ] = []

    def _sequential_gap_s(self, action: BalloonAction) -> float:
        if action.metadata and action.metadata.get("super_g_active", False):
            gap_range = SUPER_G_CLICK_GAP_RANGE_S
        elif action.metadata and action.metadata.get("g_precursor", False):
            gap_range = G_PRECURSOR_CLICK_GAP_RANGE_S
        elif action.target_type.value == "multicolor_special":
            gap_range = MULTICOLOR_SPECIAL_CLICK_GAP_RANGE_S
        else:
            gap_range = NORMAL_CLICK_GAP_RANGE_S
        return self._rng.uniform(*gap_range)

    def _twin_gap_s(self) -> float:
        return self._rng.uniform(*TWIN_CLICK_GAP_RANGE_S)

    def tick(
        self,
        decision,
        context: ExecutionContext,
        input_backend: InputBackend,
    ) -> BalloonExecutionResult:
        now = self._clock()
        if self._diagnostic:
            self._diagnostic.log(
                "executor_frame",
                now=now,
                actions=[self._action_info(a) for a in decision.actions],
                g_action=self._action_info(decision.g_action),
                notifications=[self._action_info(a) for a in decision.notifications],
                multicolor_specials=[
                    self._action_info(a) for a in decision.multicolor_specials
                ],
                input_enabled=context.input_enabled,
                active_primary=None,
                active_twin=None,
                active_g=None,
                idle_until=0.0,
            )

        if not context.input_enabled:
            return BalloonExecutionResult()

        clicks = 0
        input_timestamp = None
        action_timestamps = None
        twin_completion = None
        g_completed = False
        completed_primary_action = None
        outcomes: list[SessionOutcome] = []

        g_action = decision.g_action
        if g_action is not None:
            g_screen = self._to_screen(
                g_action.position,
                context.region_left,
                context.region_top,
            )
            g_behavior = self._fast_behavioral_action(
                g_screen,
                g_action.target_width,
            )
            g_ts = self._fast_click(input_backend, g_screen)
            self._log_click_timing(
                g_ts,
                lane="g",
                target=g_action,
                click_kind=self._click_kind(g_action),
            )
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

        actions = decision.actions
        if not actions:
            return BalloonExecutionResult(
                clicks_sent=clicks,
                input_timestamp=input_timestamp,
            )

        self._prune_recent_fast_clicks()
        seen_ids: set[int] = set()
        dispatched_primary_actions = 0

        for action in actions:
            # The vision/decision layer can surface several visible targets in
            # the same frame. Dispatch them sequentially using the timing
            # range associated with the target category.
            if isinstance(action, TwinAction):
                first, second = action.targets
                pair_ids = (first.target_id, second.target_id)
                if any(
                    target_id is not None and target_id in seen_ids
                    for target_id in pair_ids
                ):
                    continue

                first_xy = self._to_screen(
                    first.position,
                    context.region_left,
                    context.region_top,
                )
                second_xy = self._to_screen(
                    second.position,
                    context.region_left,
                    context.region_top,
                )
                if dispatched_primary_actions:
                    time.sleep(self._sequential_gap_s(first))

                first_behavior = self._fast_behavioral_action(
                    first_xy,
                    first.target_width,
                )
                second_behavior = self._fast_behavioral_action(
                    second_xy,
                    second.target_width,
                )
                first_down, first_up, second_down, second_up = input_backend.click_pair(
                    first_xy,
                    second_xy,
                    self._twin_gap_s(),
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
                self._log_click_timing(
                    first_ts,
                    lane="twin",
                    target=first,
                    click_kind="twin",
                )
                self._log_click_timing(
                    second_ts,
                    lane="twin",
                    target=second,
                    click_kind="twin",
                )
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
                dispatched_primary_actions += 1
                seen_ids.update(
                    target_id for target_id in pair_ids if target_id is not None
                )
                continue

            if action.target_id is not None and action.target_id in seen_ids:
                continue

            if (
                action.target_type.value == "normal_balloon"
                and self._was_recently_fast_clicked(action)
            ):
                if self._diagnostic:
                    self._diagnostic.log(
                        "fast_click_suppressed",
                        lane="primary",
                        target=self._action_info(action),
                        reason="recent_normal_target_click",
                    )
                continue

            if dispatched_primary_actions:
                time.sleep(self._sequential_gap_s(action))

            screen_xy = self._to_screen(
                action.position,
                context.region_left,
                context.region_top,
            )
            click_xy = self._normal_click_offset(screen_xy)
            behavioral_action = self._fast_behavioral_action(
                click_xy,
                action.target_width,
            )
            action_timestamps = self._fast_click(input_backend, click_xy)
            self._log_click_timing(
                action_timestamps,
                lane="primary",
                target=action,
                click_kind=self._click_kind(action),
            )
            clicks += 1
            input_timestamp = action_timestamps.click_up
            completed_primary_action = action
            dispatched_primary_actions += 1
            outcomes.append(
                self._make_outcome(
                    behavioral_action,
                    action_timestamps,
                    click_xy[0],
                    click_xy[1],
                    action.target_type.value,
                    target_id=action.target_id,
                    execution_status="fast",
                )
            )
            if action.target_id is not None:
                seen_ids.add(action.target_id)
            if action.target_type.value == "normal_balloon":
                self._record_fast_click(action)
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

    def shutdown(
        self,
        input_backend: InputBackend,
        reason: str,
    ) -> tuple[SessionOutcome, ...]:
        if self._diagnostic:
            self._diagnostic.log(
                "session_shutdown",
                reason=reason,
                interrupted_actions=0,
            )
        return ()

    def _prune_recent_fast_clicks(self) -> None:
        cutoff = self._clock() - 0.200
        self._recent_fast_clicks = [
            entry for entry in self._recent_fast_clicks if entry[0] >= cutoff
        ]

    def _was_recently_fast_clicked(self, action: BalloonAction) -> bool:
        if action.target_type.value != "normal_balloon":
            return False
        now = self._clock()
        ax, ay = action.position
        radius = max(12.0, float(action.target_width) * 0.5)
        radius_sq = radius * radius
        for clicked_at, target_id, (cx, cy), clicked_radius in self._recent_fast_clicks:
            if now - clicked_at > 0.200:
                continue
            if target_id is not None and action.target_id == target_id:
                return True
            combined_radius = max(radius, clicked_radius)
            dx = ax - cx
            dy = ay - cy
            if dx * dx + dy * dy <= combined_radius * combined_radius:
                return True
        return False

    def _record_fast_click(self, action: BalloonAction) -> None:
        self._recent_fast_clicks.append(
            (
                self._clock(),
                action.target_id,
                action.position,
                max(12.0, float(action.target_width) * 0.5),
            )
        )

    def _normal_click_offset(self, screen_xy: tuple[int, int]) -> tuple[int, int]:
        """Apply a tiny randomized endpoint offset to normal-balloon clicks only."""
        dx = self._rng.randint(*NORMAL_CLICK_OFFSET_PX_RANGE)
        dy = self._rng.randint(*NORMAL_CLICK_OFFSET_PX_RANGE)
        return screen_xy[0] + dx, screen_xy[1] + dy

    def _fast_click(
        self,
        input_backend: InputBackend,
        screen_xy: tuple[int, int],
    ) -> ActionTimestamps:
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

    def _log_click_timing(
        self,
        timestamps: ActionTimestamps,
        *,
        lane: str,
        target: BalloonAction,
        click_kind: str,
    ) -> None:
        if not self._diagnostic:
            return

        click_down = timestamps.click_down
        click_up = timestamps.click_up
        if click_down is None or click_up is None:
            return

        previous_click_down = self._last_click_down_ts
        inter_click_gap_ms = (
            max(0.0, (click_down - previous_click_down) * 1000.0)
            if previous_click_down is not None
            else None
        )
        movement_ms = max(
            0.0,
            (timestamps.movement_end - timestamps.movement_start) * 1000.0,
        )
        hold_ms = max(0.0, (click_up - click_down) * 1000.0)
        input_ms = max(
            0.0,
            (click_up - timestamps.action_planned) * 1000.0,
        )

        self._diagnostic.log(
            "click_timing",
            lane=lane,
            click_kind=click_kind,
            target=self._action_info(target),
            action_planned_ts=timestamps.action_planned,
            movement_start_ts=timestamps.movement_start,
            movement_end_ts=timestamps.movement_end,
            click_down_ts=click_down,
            click_up_ts=click_up,
            movement_ms=movement_ms,
            click_hold_ms=hold_ms,
            input_total_ms=input_ms,
            inter_click_down_gap_ms=inter_click_gap_ms,
        )
        self._last_click_down_ts = click_down

    @staticmethod
    def _click_kind(action: BalloonAction) -> str:
        metadata = action.metadata or {}
        if metadata.get("super_g_active", False):
            return "super_g"
        if metadata.get("g_precursor", False):
            return "g_precursor"
        return action.target_type.value

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
                (timestamps.click_down - behavioral_action.acquisition_timestamp)
                / (1000.0 / 60.0),
            )
            estimated_click_tick = observed_reaction
            timing_error_ms = (
                observed_reaction - behavioral_action.intended_reaction_ticks
            ) * (1000.0 / 60.0)
        scheduler_delay_ms = max(
            0.0,
            (timestamps.reaction_end - timestamps.action_planned) * 1000.0
            - behavioral_action.reaction_time_s * 1000.0,
        )
        movement_duration_ms = max(
            0.0,
            (timestamps.movement_end - timestamps.movement_start) * 1000.0,
        )
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
            click_tick_modulo=(
                int(round(estimated_click_tick)) % 3
                if estimated_click_tick is not None
                else None
            ),
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
            observation_notes=[
                "UNOBSERVABLE THROUGH CURRENT INTERFACE: authoritative spawnTick/popTick/hit state; click tick is a relative estimate from first-seen"
            ],
        )

    @staticmethod
    def _action_info(action):
        if action is None:
            return None
        if isinstance(action, TwinAction):
            return {
                "type": "TwinAction",
                "targets": [FastExecutionPolicy._action_info(t) for t in action.targets],
            }
        return {
            "type": action.target_type.value,
            "id": action.target_id,
            "color": action.color.value if action.color else None,
            "position": list(action.position),
            "width": action.target_width,
        }

    @staticmethod
    def _to_screen(position, region_left, region_top):
        return frame_to_screen(position[0], position[1], region_left, region_top)
