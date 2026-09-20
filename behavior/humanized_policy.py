from __future__ import annotations

from dataclasses import replace
import math
import random
import time

from behavior.fast_policy import FastExecutionPolicy
from behavior.action_scheduler import ActionTimestamps, TwinActionCompletion
from diagnostics.downstream import DownstreamDiagnostic
from game_profiles.balloon_types import BalloonAction, TargetType, TwinAction
from telemetry.outcome import SessionOutcome
from behavior.profile import HumanProfile
from .execution_policy import (
    BalloonExecutionResult,
    ExecutionContext,
    ExecutionPolicy,
    InputBackend,
)


class _DelayedTwinInputBackend:
    """Delegating input backend that adds only the requested duo inter-click delay."""

    def __init__(self, backend: InputBackend, inter_click_delay_s: float) -> None:
        self._backend = backend
        self._inter_click_delay_s = inter_click_delay_s

    def move_to(self, x: int, y: int):
        return self._backend.move_to(x, y)

    def left_down(self):
        return self._backend.left_down()

    def left_up(self):
        return self._backend.left_up()

    def click_pair(
        self,
        first: tuple[int, int],
        second: tuple[int, int],
        inter_click_delay_s: float = 0.0,
    ) -> tuple[float, float, float, float]:
        first_move = self._backend.move_to(*first)
        first_down = self._backend.left_down()
        first_up = self._backend.left_up()
        time.sleep(self._inter_click_delay_s)
        self._backend.move_to(*second)
        second_down = self._backend.left_down()
        second_up = self._backend.left_up()
        return first_down, first_up, second_down, second_up


class HumanizedExecutionPolicy:
    """Humanized execution at the execution boundary.

    Vision, tracking, decision mechanics, and the fast policy remain unchanged.
    Normal endpoints receive radius-safe spatial offsets. Reaction delays are
    sampled per target type and applied before dispatch. The actual G target
    keeps the old center-click behavior.
    """

    _OFFSET_SIGMA_PX = 5.0
    _RADIUS_SAFETY_MARGIN_PX = 1.0
    _GAUSSIAN_Z_99 = 2.576
    _ENGINE_TICK_MS = 1000.0 / 60.0
    _COMBO_MAX_DELAY_MS = 400.0

    # Explicit timing walls derived from the game engine.
    _REACTION_BOUNDS_MS = {
        "standard": (150.0, 800.0),
        "notification": (150.0, 800.0),
        "golden_duo_first": (150.0, 300.0),
        "golden_duo_second": (150.0, 300.0),
        "gemini": (150.0, 1500.0),
        "super_g_node": (150.0, 800.0),
        "balloon_g": (150.0, 300.0),
    }

    def __init__(
        self,
        diagnostic: DownstreamDiagnostic | None = None,
        *,
        rng: random.Random | None = None,
        reaction_rng: random.Random | None = None,
    ) -> None:
        self._diagnostic = diagnostic
        self._rng = rng or random.Random()
        self._reaction_rng = reaction_rng or random.Random()
        self._profile: HumanProfile | None = None
        self._last_reaction_ticks: float | None = None
        self._session_started_at: float | None = None
        self._fast = FastExecutionPolicy(
            diagnostic=diagnostic,
        )

    def set_profile(self, profile: HumanProfile) -> None:
        """Bind the immutable per-session profile used by reaction/offset sampling."""
        self._profile = profile
        self._last_reaction_ticks = profile.reaction_mu_ticks
        self._session_started_at = None

    def tick(
        self,
        decision,
        context: ExecutionContext,
        input_backend: InputBackend,
    ) -> BalloonExecutionResult:
        if not context.input_enabled:
            return BalloonExecutionResult()

        if self._session_started_at is None:
            self._session_started_at = time.perf_counter()

        combined = BalloonExecutionResult()

        # Preserve FastExecutionPolicy's G-first ordering, but give the actual
        # G target its target-specific reaction delay while keeping its center.
        if decision.g_action is not None:
            delay_s = self._reaction_delay_s(
                decision.g_action.target_type,
                is_twin_second=False,
                combo_eligible=False,
            )
            time.sleep(delay_s)
            g_decision = replace(
                decision,
                actions=(),
                g_action=decision.g_action,
                multicolor_specials=(),
                notifications=(),
            )
            g_result = self._fast.tick(g_decision, context, input_backend)
            combined = self._combine(combined, g_result)

        for action in decision.actions:
            if isinstance(action, TwinAction):
                first = self._offset_balloon(action.targets[0])
                second = self._offset_balloon(action.targets[1])

                # The duo wall applies to the pair delta, not just reaction
                # time to the first click. Keep both sampled delays inside
                # 150–300 ms, so the total first->second delay is bounded by
                # 300–600 ms and remains inside the engine's 750 ms window.
                first_delay_s = self._reaction_delay_s(
                    first.target_type,
                    is_twin_second=False,
                    combo_eligible=False,
                )
                second_delay_s = self._reaction_delay_s(
                    second.target_type,
                    is_twin_second=True,
                    combo_eligible=False,
                )
                time.sleep(first_delay_s)

                twin = replace(action, targets=(first, second))
                pair_decision = replace(
                    decision,
                    actions=(twin,),
                    g_action=None,
                    multicolor_specials=(),
                    notifications=(),
                )
                pair_result = self._fast.tick(
                    pair_decision,
                    context,
                    _DelayedTwinInputBackend(
                        input_backend,
                        second_delay_s,
                    ),
                )
                combined = self._combine(combined, pair_result)
                continue

            transformed = self._offset_balloon(action)
            delay_s = self._reaction_delay_s(
                transformed.target_type,
                is_twin_second=False,
                combo_eligible=self._is_combo_eligible(transformed.target_type),
            )
            time.sleep(delay_s)
            result = self._fast_single(
                transformed,
                decision,
                context,
                input_backend,
            )
            combined = self._combine(combined, result)

        return combined

    def shutdown(
        self,
        input_backend: InputBackend,
        reason: str,
    ) -> tuple[SessionOutcome, ...]:
        return self._fast.shutdown(input_backend, reason)

    def _fast_single(
        self,
        action: BalloonAction,
        decision,
        context: ExecutionContext,
        input_backend: InputBackend,
    ) -> BalloonExecutionResult:
        single_decision = replace(
            decision,
            actions=(action,),
            g_action=None,
            multicolor_specials=(),
            notifications=(),
        )
        return self._fast.tick(single_decision, context, input_backend)

    @staticmethod
    def _combine(
        left: BalloonExecutionResult,
        right: BalloonExecutionResult,
    ) -> BalloonExecutionResult:
        return BalloonExecutionResult(
            clicks_sent=left.clicks_sent + right.clicks_sent,
            input_timestamp=right.input_timestamp or left.input_timestamp,
            action_timestamps=right.action_timestamps or left.action_timestamps,
            twin_completion=right.twin_completion or left.twin_completion,
            g_click_completed=left.g_click_completed or right.g_click_completed,
            completed_primary_action=right.completed_primary_action or left.completed_primary_action,
            outcomes=left.outcomes + right.outcomes,
        )

    @staticmethod
    def _is_combo_eligible(target_type: TargetType) -> bool:
        return target_type in {TargetType.NORMAL_BALLOON, TargetType.SPECIAL_SEQUENCE_BALLOON}

    def _reaction_delay_s(
        self,
        target_type: TargetType,
        *,
        is_twin_second: bool,
        combo_eligible: bool,
    ) -> float:
        key = self._reaction_key(target_type, is_twin_second=is_twin_second)
        lower_ms, upper_ms = self._REACTION_BOUNDS_MS[key]

        # Midpoint gives the requested explicit center of each timing wall.
        # Sigma is chosen so a symmetric Gaussian puts ~99% of its mass inside
        # the bounds; the final draw is clipped defensively.
        median_ms = (lower_ms + upper_ms) * 0.5
        sigma_ms = min(median_ms - lower_ms, upper_ms - median_ms) / self._GAUSSIAN_Z_99
        sampled_ms = self._reaction_rng.gauss(median_ms, sigma_ms)
        sampled_ms = max(lower_ms, min(upper_ms, sampled_ms))

        sampled_ticks = sampled_ms / self._ENGINE_TICK_MS
        profile = self._profile
        ar1 = profile.reaction_ar1 if profile is not None else 0.30
        previous_ticks = (
            self._last_reaction_ticks
            if self._last_reaction_ticks is not None
            else sampled_ticks
        )
        correlated_ticks = ar1 * previous_ticks + (1.0 - ar1) * sampled_ticks

        # Preserve the existing profile's perception contribution and session
        # modifiers, then enforce the target-specific timing wall again.
        if profile is not None:
            correlated_ticks += profile.perception_ticks
            elapsed = (
                max(0.0, time.perf_counter() - self._session_started_at)
                if self._session_started_at is not None
                else 0.0
            )
            warmup_factor = (
                max(0.0, profile.warmup_sec - elapsed)
                / max(0.001, profile.warmup_sec)
                * 0.15
            )
            fatigue_factor = min(
                0.30,
                max(0.0, elapsed - 300.0)
                / 60.0
                * profile.fatigue_per_min
                / 100.0,
            )
        else:
            warmup_factor = 0.0
            fatigue_factor = 0.0

        actual_ms = (
            correlated_ticks
            * self._ENGINE_TICK_MS
            * (1.0 + warmup_factor + fatigue_factor)
        )
        effective_upper_ms = min(
            upper_ms,
            self._COMBO_MAX_DELAY_MS if combo_eligible else upper_ms,
        )
        actual_ms = max(lower_ms, min(effective_upper_ms, actual_ms))
        self._last_reaction_ticks = actual_ms / self._ENGINE_TICK_MS

        if self._diagnostic:
            self._diagnostic.log(
                "humanized_reaction",
                target_type=target_type.value,
                reaction_key=key,
                lower_ms=lower_ms,
                upper_ms=upper_ms,
                effective_upper_ms=effective_upper_ms,
                sampled_ms=sampled_ms,
                gaussian_median_ms=median_ms,
                gaussian_sigma_ms=sigma_ms,
                ar1=ar1,
                ar1_applied_to="reaction_delay",
                warmup_factor=warmup_factor,
                fatigue_factor=fatigue_factor,
                final_delay_ms=actual_ms,
            )

        return actual_ms / 1000.0

    @staticmethod
    def _reaction_key(
        target_type: TargetType,
        *,
        is_twin_second: bool,
    ) -> str:
        if target_type == TargetType.G_SYMBOL:
            return "balloon_g"
        if target_type == TargetType.MULTICOLOR_SPECIAL:
            return "gemini"
        if target_type == TargetType.NOTIFICATION:
            return "notification"
        if target_type == TargetType.TWIN_BALLOON:
            return "golden_duo_second" if is_twin_second else "golden_duo_first"
        if target_type == TargetType.SPECIAL_SEQUENCE_BALLOON:
            return "super_g_node"
        return "standard"

    def _offset_action(self, action: BalloonAction | TwinAction):
        if isinstance(action, TwinAction):
            first = self._offset_balloon(action.targets[0])
            second = self._offset_balloon(action.targets[1])
            return replace(action, targets=(first, second))
        return self._offset_balloon(action)

    def _offset_balloon(self, action: BalloonAction) -> BalloonAction:
        # Actual G remains center-clicked; this preserves the known-working
        # behavior after ML/tracker localization uncertainty.
        if action.target_type == TargetType.G_SYMBOL:
            return action

        radius = max(0.0, float(action.target_width) * 0.5)
        max_offset = max(0.0, radius - self._RADIUS_SAFETY_MARGIN_PX)

        sigma = (
            self._profile.spatial_sigma_px
            if self._profile is not None
            else self._OFFSET_SIGMA_PX
        )
        dx = self._rng.gauss(0.0, sigma)
        dy = self._rng.gauss(0.0, sigma)
        magnitude = math.hypot(dx, dy)
        clipped = magnitude > max_offset and max_offset > 0.0
        if clipped:
            scale = max_offset / magnitude
            dx *= scale
            dy *= scale
        elif max_offset <= 0.0:
            dx = 0.0
            dy = 0.0

        center_x, center_y = action.position
        dispatch_x = int(round(center_x + dx))
        dispatch_y = int(round(center_y + dy))

        actual_dx = dispatch_x - center_x
        actual_dy = dispatch_y - center_y
        actual_magnitude = math.hypot(actual_dx, actual_dy)
        if actual_magnitude > max_offset and actual_magnitude > 0.0:
            scale = max_offset / actual_magnitude
            dispatch_x = int(round(center_x + actual_dx * scale))
            dispatch_y = int(round(center_y + actual_dy * scale))

        offset_x = dispatch_x - center_x
        offset_y = dispatch_y - center_y
        if self._diagnostic:
            self._diagnostic.log(
                "humanized_offset",
                target_id=action.target_id,
                target_type=action.target_type.value,
                center=[center_x, center_y],
                dispatch=[dispatch_x, dispatch_y],
                offset=[offset_x, offset_y],
                offset_px=math.hypot(offset_x, offset_y),
                sampled_offset_px=magnitude,
                profile_spatial_sigma_px=sigma,
                target_radius_px=radius,
                safe_radius_px=max_offset,
                clipped=clipped or (
                    offset_x != round(dx) or offset_y != round(dy)
                ),
            )

        return replace(action, position=(dispatch_x, dispatch_y))
