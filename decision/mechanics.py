from __future__ import annotations

from dataclasses import dataclass
from math import hypot
from typing import Any

from config.settings import Settings
from game_profiles.balloon_types import (
    BalloonAction,
    BalloonColor,
    BalloonGameState,
    GameMode,
    SpecialSequenceStep,
    TargetType,
    TwinAction,
)


@dataclass(frozen=True)
class DecisionResult:
    """Decision output split into primary gameplay and concurrent mechanics."""

    actions: tuple[BalloonAction | TwinAction, ...]
    g_action: BalloonAction | None
    multicolor_specials: tuple[BalloonAction, ...]
    mode: GameMode
    special_step: SpecialSequenceStep


class BalloonDecisionEngine:
    """Apply balloon-game rules to structured vision detections.

    Vision remains responsible only for describing what is visible. This class
    owns the one-time special sequence, twin pairing, and concurrent G state.
    """

    _SPECIAL_ORDER = (
        BalloonColor.BLUE,
        BalloonColor.RED,
        BalloonColor.YELLOW,
        BalloonColor.GREEN,
    )

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self.state = BalloonGameState()

    def reset(self) -> None:
        self.state = BalloonGameState()

    def decide(self, detections: list[dict[str, Any]]) -> DecisionResult:
        visible = [
            d for d in detections
            if d.get("target_type") != TargetType.INVALID_WHITE.value
        ]

        g_detections = [
            d for d in visible
            if d.get("target_type") == TargetType.G_SYMBOL.value
        ]
        g_action = self._make_action(g_detections[0], is_g=True) if g_detections else None
        self.state.g_active = g_action is not None

        multicolor = [
            self._make_action(d)
            for d in visible
            if d.get("target_type") == TargetType.MULTICOLOR_SPECIAL.value
        ]

        special = [
            d for d in visible
            if d.get("target_type") == TargetType.SPECIAL_SEQUENCE_BALLOON.value
        ]
        if (
            self.state.mode == GameMode.NORMAL_PLAY
            and not self.state.special_sequence_seen
            and special
        ):
            self.state.mode = GameMode.SPECIAL_SEQUENCE
            self.state.special_step = SpecialSequenceStep.BLUE
            self.state.special_sequence_seen = True

        if self.state.mode == GameMode.SPECIAL_SEQUENCE:
            actions = self._select_required_special(special)
        else:
            actions = self._select_normal_play(visible)

        return DecisionResult(
            actions=tuple(actions),
            g_action=g_action,
            multicolor_specials=tuple(multicolor),
            mode=self.state.mode,
            special_step=self.state.special_step,
        )

    def complete_action(self, action: BalloonAction) -> bool:
        """Advance the special sequence only after the required target completes."""
        if (
            self.state.mode != GameMode.SPECIAL_SEQUENCE
            or action.target_type != TargetType.SPECIAL_SEQUENCE_BALLOON
            or action.color != self.state.required_special_color
        ):
            return False

        step_index = self._SPECIAL_ORDER.index(action.color)
        if step_index == len(self._SPECIAL_ORDER) - 1:
            self.state.mode = GameMode.NORMAL_PLAY
            self.state.special_step = SpecialSequenceStep.COMPLETE
            return True

        self.state.special_step = SpecialSequenceStep(
            self._SPECIAL_ORDER[step_index + 1].value
        )
        return True

    def _select_required_special(
        self,
        detections: list[dict[str, Any]],
    ) -> list[BalloonAction]:
        required = self.state.required_special_color
        if required is None:
            return []

        for detection in detections:
            if detection.get("color_name") == required.value:
                return [self._make_action(detection, is_special=True)]
        return []

    def _select_normal_play(
        self,
        detections: list[dict[str, Any]],
    ) -> list[BalloonAction | TwinAction]:
        normal: list[tuple[int, dict[str, Any]]] = []
        twins: list[tuple[int, dict[str, Any]]] = []

        for index, detection in enumerate(detections):
            target_type = detection.get("target_type")
            if target_type == TargetType.NORMAL_BALLOON.value:
                normal.append((index, detection))
            elif target_type == TargetType.TWIN_BALLOON.value:
                if detection.get("color_name") in {
                    BalloonColor.YELLOW.value,
                    BalloonColor.GREEN.value,
                }:
                    twins.append((index, detection))

        pair = self._pair_twins(twins)
        paired_indices = {index for index, _ in pair} if pair else set()

        indexed_actions: list[tuple[int, BalloonAction | TwinAction]] = []
        if pair:
            first_index, first_detection = pair[0]
            second_index, second_detection = pair[1]
            first = self._make_action(first_detection, is_twin=True)
            second = self._make_action(second_detection, is_twin=True)
            indexed_actions.append((
                min(first_index, second_index),
                TwinAction(
                    targets=(first, second),
                    max_click_delta_ms=30.0,
                    metadata={"max_click_delta_ms": 30.0},
                ),
            ))

        for index, detection in normal:
            indexed_actions.append((index, self._make_action(detection)))

        indexed_actions.sort(key=lambda item: item[0])
        return [action for _, action in indexed_actions]

    def _pair_twins(
        self,
        twins: list[tuple[int, dict[str, Any]]],
    ) -> tuple[tuple[int, dict[str, Any]], tuple[int, dict[str, Any]]] | None:
        if len(twins) != 2:
            return None

        (_, first), (_, second) = twins
        if first.get("color_name") != second.get("color_name"):
            return None

        a = first.get("center")
        b = second.get("center")
        if not a or not b:
            return None

        max_distance = float(self._settings.grouping.max_pair_distance)
        if hypot(a[0] - b[0], a[1] - b[1]) > max_distance:
            return None

        return twins[0], twins[1]

    @staticmethod
    def _make_action(
        detection: dict[str, Any],
        *,
        is_twin: bool = False,
        is_special: bool = False,
        is_g: bool = False,
    ) -> BalloonAction:
        target_type = TargetType(detection["target_type"])
        color_name = detection.get("color_name")
        color = BalloonColor(color_name) if color_name else None
        return BalloonAction(
            target_id=detection.get("target_id"),
            target_type=target_type,
            color=color,
            position=tuple(detection.get("center", (0, 0))),
            target_width=float(detection.get("width", 0.0)),
            is_twin=is_twin,
            is_special_sequence=is_special,
            is_g=is_g,
            metadata={
                "confidence": detection.get("color_confidence", detection.get("confidence", 0.0)),
                "bbox": detection.get("bbox"),
            },
        )
