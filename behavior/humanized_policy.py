from __future__ import annotations

from dataclasses import replace
import math
import random

from behavior.fast_policy import FastExecutionPolicy
from diagnostics.downstream import DownstreamDiagnostic
from game_profiles.balloon_types import BalloonAction, TwinAction
from telemetry.outcome import SessionOutcome
from behavior.profile import HumanProfile
from .execution_policy import (
    BalloonExecutionResult,
    ExecutionContext,
    ExecutionPolicy,
    InputBackend,
)


class HumanizedExecutionPolicy:
    """Fast execution with only a small, radius-safe endpoint offset.

    Phase B deliberately stays at the execution boundary: vision, tracking,
    decision mechanics, cadence, and G-lane mechanics remain unchanged.
    """

    _OFFSET_SIGMA_PX = 5.0
    _RADIUS_SAFETY_MARGIN_PX = 1.0

    def __init__(
        self,
        diagnostic: DownstreamDiagnostic | None = None,
        *,
        rng: random.Random | None = None,
    ) -> None:
        self._diagnostic = diagnostic
        self._rng = rng or random.Random()
        self._profile: HumanProfile | None = None
        self._fast = FastExecutionPolicy(
            diagnostic=diagnostic,
        )

    def set_profile(self, profile: HumanProfile) -> None:
        """Bind the immutable per-session profile used by endpoint sampling."""
        self._profile = profile

    def tick(
        self,
        decision,
        context: ExecutionContext,
        input_backend: InputBackend,
    ) -> BalloonExecutionResult:
        if not context.input_enabled:
            return BalloonExecutionResult()

        transformed = replace(
            decision,
            actions=tuple(self._offset_action(action) for action in decision.actions),
            g_action=(
                self._offset_action(decision.g_action)
                if decision.g_action is not None
                else None
            ),
            multicolor_specials=tuple(
                self._offset_action(action)
                for action in decision.multicolor_specials
            ),
            notifications=tuple(
                self._offset_action(action)
                for action in decision.notifications
            ),
        )
        return self._fast.tick(transformed, context, input_backend)

    def shutdown(
        self,
        input_backend: InputBackend,
        reason: str,
    ) -> tuple[SessionOutcome, ...]:
        return self._fast.shutdown(input_backend, reason)

    def _offset_action(self, action: BalloonAction | TwinAction):
        if isinstance(action, TwinAction):
            first = self._offset_balloon(action.targets[0])
            second = self._offset_balloon(action.targets[1])
            return replace(action, targets=(first, second))
        return self._offset_balloon(action)

    def _offset_balloon(self, action: BalloonAction) -> BalloonAction:
        radius = max(0.0, float(action.target_width) * 0.5)
        max_offset = max(0.0, radius - self._RADIUS_SAFETY_MARGIN_PX)

        sigma = self._profile.spatial_sigma_px if self._profile is not None else self._OFFSET_SIGMA_PX
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

        # Integer rounding can add a fraction of a pixel. Keep the actual
        # dispatched coordinate inside the conservative radius margin.
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
                clipped=clipped or (offset_x != round(dx) or offset_y != round(dy)),
            )

        # The transformed position is the coordinate that FastExecutionPolicy
        # hands to move_to(), so left_down() occurs at this exact OS cursor
        # location. target_width is preserved, keeping the current-frame
        # hitbox/radius available for the safety clamp.
        return replace(action, position=(dispatch_x, dispatch_y))
