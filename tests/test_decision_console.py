from __future__ import annotations

import io
import unittest
from contextlib import redirect_stdout
from types import SimpleNamespace

from diagnostics.decision_console import DecisionConsole
from game_profiles.balloon_types import (
    BalloonAction,
    BalloonColor,
    GameMode,
    SpecialSequenceStep,
    TargetType,
    TwinAction,
)


def action(target_type, color=None, target_id=1, position=(100, 200)):
    return BalloonAction(
        target_id=target_id,
        target_type=target_type,
        color=color,
        position=position,
        target_width=30.0,
    )


class TestDecisionConsole(unittest.TestCase):
    def test_disabled_console_is_silent(self):
        console = DecisionConsole(False)
        decision = SimpleNamespace(
            actions=(),
            g_action=None,
            multicolor_specials=(),
            notifications=(),
            mode=GameMode.NORMAL_PLAY,
            special_step=SpecialSequenceStep.COMPLETE,
        )
        out = io.StringIO()
        with redirect_stdout(out):
            console.frame(1, [], decision)
        self.assertEqual(out.getvalue(), "")

    def test_frame_prints_detection_rejection_and_click(self):
        console = DecisionConsole(True)
        normal = action(
            TargetType.NORMAL_BALLOON,
            BalloonColor.BLUE,
            target_id=7,
            position=(1033, 778),
        )
        detections = [{
            "target_type": TargetType.NORMAL_BALLOON.value,
            "color_name": "blue",
            "target_id": 7,
            "center": (1033, 778),
            "metadata": {
                "temporal": {"static_ui_suspect": False},
                "click_attempts": 0,
            },
        }, {
            "target_type": TargetType.NORMAL_BALLOON.value,
            "color_name": "blue",
            "target_id": 8,
            "center": (1005, 261),
            "metadata": {
                "temporal": {"static_ui_suspect": True},
                "click_attempts": 0,
            },
        }]
        decision = SimpleNamespace(
            actions=(normal,),
            g_action=None,
            multicolor_specials=(),
            notifications=(),
            mode=GameMode.NORMAL_PLAY,
            special_step=SpecialSequenceStep.COMPLETE,
        )

        out = io.StringIO()
        with redirect_stdout(out):
            console.frame(12, detections, decision)

        text = out.getvalue()
        self.assertIn("DETECTED:", text)
        self.assertIn("blue normal_balloon#7@(1033, 778)", text)
        self.assertIn("REJECTED:", text)
        self.assertIn("static_ui_suspect", text)
        self.assertIn("CLICK: blue normal_balloon#7@(1033, 778)", text)

    def test_twin_and_super_g_are_explicit(self):
        console = DecisionConsole(True)
        first = action(TargetType.TWIN_BALLOON, BalloonColor.YELLOW, 20, (100, 100))
        second = action(TargetType.TWIN_BALLOON, BalloonColor.YELLOW, 21, (150, 100))
        twin = TwinAction(targets=(first, second))
        decision = SimpleNamespace(
            actions=(twin,),
            g_action=None,
            multicolor_specials=(),
            notifications=(),
            mode=GameMode.SPECIAL_SEQUENCE,
            special_step=SpecialSequenceStep.RED,
        )

        out = io.StringIO()
        with redirect_stdout(out):
            console.frame(25, [], decision)

        text = out.getvalue()
        self.assertIn("TWIN: CLICK", text)
        self.assertIn("twin_balloon#20", text)
        self.assertIn("twin_balloon#21", text)
        self.assertIn("SUPER-G: required=red", text)

    def test_execution_reports_actual_click_outcome(self):
        console = DecisionConsole(True)
        outcome = SimpleNamespace(
            target_type="normal_balloon",
            target_id=7,
            execution_status="completed",
            screen_click_x=1033,
            screen_click_y=778,
        )
        execution = SimpleNamespace(
            clicks_sent=1,
            outcomes=(outcome,),
            completed_primary_action=None,
            twin_completion=None,
            g_click_completed=False,
        )

        out = io.StringIO()
        with redirect_stdout(out):
            console.execution(12, execution)

        text = out.getvalue()
        self.assertIn("[EXEC 12] SENT 1 CLICK", text)
        self.assertIn("normal_balloon#7 @ (1033,778)", text)
        self.assertIn("status=completed", text)


if __name__ == "__main__":
    unittest.main()
