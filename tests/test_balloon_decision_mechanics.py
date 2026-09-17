from __future__ import annotations

import unittest

from config.settings import Settings
from decision.mechanics import BalloonDecisionEngine
from game_profiles.balloon_types import (
    BalloonColor,
    GameMode,
    SpecialSequenceStep,
    TargetType,
    TwinAction,
)


def detection(
    target_type: TargetType,
    color: BalloonColor | None,
    center: tuple[int, int],
    *,
    target_id: int | None = None,
) -> dict:
    return {
        "target_type": target_type.value,
        "color_name": color.value if color else None,
        "center": center,
        "bbox": (center[0] - 20, center[1] - 20, 40, 40),
        "area": 1600,
        "color_confidence": 0.95,
        "confidence": 0.95,
        "width": 40.0,
        "height": 40.0,
        "target_id": target_id,
    }


class TestBalloonDecisionMechanics(unittest.TestCase):
    def setUp(self) -> None:
        self.settings = Settings()
        self.settings.grouping.max_pair_distance = 200
        self.engine = BalloonDecisionEngine(self.settings)

    def test_normal_play_selects_normal_targets_without_color_priority(self):
        detections = [
            detection(TargetType.NORMAL_BALLOON, BalloonColor.GREEN, (100, 100), target_id=1),
            detection(TargetType.NORMAL_BALLOON, BalloonColor.BLUE, (200, 100), target_id=2),
        ]
        result = self.engine.decide(detections)

        self.assertEqual(result.mode, GameMode.NORMAL_PLAY)
        self.assertEqual([a.color for a in result.actions], [
            BalloonColor.GREEN,
            BalloonColor.BLUE,
        ])
        self.assertFalse(any(a.is_special_sequence for a in result.actions))

    def test_special_detection_enters_strict_blue_step(self):
        detections = [
            detection(TargetType.NORMAL_BALLOON, BalloonColor.RED, (100, 100), target_id=1),
            detection(TargetType.SPECIAL_SEQUENCE_BALLOON, BalloonColor.BLUE, (300, 100), target_id=2),
        ]
        result = self.engine.decide(detections)

        self.assertEqual(result.mode, GameMode.SPECIAL_SEQUENCE)
        self.assertEqual(result.special_step, SpecialSequenceStep.BLUE)
        self.assertEqual(len(result.actions), 1)
        self.assertEqual(result.actions[0].target_type, TargetType.SPECIAL_SEQUENCE_BALLOON)
        self.assertEqual(result.actions[0].color, BalloonColor.BLUE)

    def test_special_sequence_ignores_ordinary_same_color_and_twins(self):
        result = self.engine.decide([
            detection(TargetType.SPECIAL_SEQUENCE_BALLOON, BalloonColor.BLUE, (500, 100), target_id=10),
            detection(TargetType.NORMAL_BALLOON, BalloonColor.BLUE, (100, 100), target_id=1),
            detection(TargetType.TWIN_BALLOON, BalloonColor.BLUE, (200, 100), target_id=2),
            detection(TargetType.TWIN_BALLOON, BalloonColor.BLUE, (240, 100), target_id=3),
        ])

        self.assertEqual(len(result.actions), 1)
        self.assertEqual(result.actions[0].target_id, 10)

        # A red special is visible early, but cannot advance the blue step.
        result = self.engine.decide([
            detection(TargetType.SPECIAL_SEQUENCE_BALLOON, BalloonColor.RED, (500, 100), target_id=11),
        ])
        self.assertEqual(result.actions, ())
        self.assertEqual(self.engine.state.special_step, SpecialSequenceStep.BLUE)

    def test_special_sequence_advances_only_on_completed_required_target(self):
        blue = self.engine.decide([
            detection(TargetType.SPECIAL_SEQUENCE_BALLOON, BalloonColor.BLUE, (500, 100), target_id=10),
        ]).actions[0]
        self.assertTrue(self.engine.complete_action(blue))
        self.assertEqual(self.engine.state.special_step, SpecialSequenceStep.RED)

        red = self.engine.decide([
            detection(TargetType.SPECIAL_SEQUENCE_BALLOON, BalloonColor.RED, (600, 100), target_id=11),
        ]).actions[0]
        self.assertTrue(self.engine.complete_action(red))
        self.assertEqual(self.engine.state.special_step, SpecialSequenceStep.YELLOW)

        yellow = self.engine.decide([
            detection(TargetType.SPECIAL_SEQUENCE_BALLOON, BalloonColor.YELLOW, (700, 100), target_id=12),
        ]).actions[0]
        self.assertTrue(self.engine.complete_action(yellow))
        self.assertEqual(self.engine.state.special_step, SpecialSequenceStep.GREEN)

        green = self.engine.decide([
            detection(TargetType.SPECIAL_SEQUENCE_BALLOON, BalloonColor.GREEN, (800, 100), target_id=13),
        ]).actions[0]
        self.assertTrue(self.engine.complete_action(green))
        self.assertEqual(self.engine.state.mode, GameMode.NORMAL_PLAY)
        self.assertEqual(self.engine.state.special_step, SpecialSequenceStep.COMPLETE)
        self.assertTrue(self.engine.state.special_sequence_seen)

    def test_completed_special_sequence_cannot_be_retriggered(self):
        for color, target_id in [
            (BalloonColor.BLUE, 90),
            (BalloonColor.RED, 91),
            (BalloonColor.YELLOW, 92),
            (BalloonColor.GREEN, 93),
        ]:
            action = self.engine.decide([
                detection(
                    TargetType.SPECIAL_SEQUENCE_BALLOON,
                    color,
                    (target_id, 100),
                    target_id=target_id,
                ),
            ]).actions[0]
            self.assertTrue(self.engine.complete_action(action))

        self.assertEqual(self.engine.state.mode, GameMode.NORMAL_PLAY)
        result = self.engine.decide([
            detection(
                TargetType.SPECIAL_SEQUENCE_BALLOON,
                BalloonColor.BLUE,
                (100, 200),
                target_id=94,
            ),
        ])
        self.assertEqual(result.mode, GameMode.NORMAL_PLAY)
        self.assertEqual(result.actions, ())

    def test_twin_pair_is_one_coordinated_action_with_30ms_bound(self):
        result = self.engine.decide([
            detection(TargetType.TWIN_BALLOON, BalloonColor.YELLOW, (100, 100), target_id=20),
            detection(TargetType.TWIN_BALLOON, BalloonColor.YELLOW, (150, 100), target_id=21),
            detection(TargetType.NORMAL_BALLOON, BalloonColor.RED, (500, 500), target_id=30),
        ])

        self.assertEqual(len(result.actions), 2)
        twin = result.actions[0]
        self.assertIsInstance(twin, TwinAction)
        self.assertEqual(
            {target.target_id for target in twin.targets},
            {20, 21},
        )
        self.assertLessEqual(twin.max_click_delta_ms, 30.0)
        self.assertEqual(result.actions[1].target_id, 30)

    def test_unpaired_twin_is_not_clicked_as_a_normal_balloon(self):
        result = self.engine.decide([
            detection(TargetType.TWIN_BALLOON, BalloonColor.GREEN, (100, 100), target_id=40),
        ])
        self.assertEqual(result.actions, ())

    def test_white_target_is_never_actionable(self):
        result = self.engine.decide([
            detection(TargetType.INVALID_WHITE, None, (100, 100), target_id=50),
            detection(TargetType.NORMAL_BALLOON, BalloonColor.RED, (200, 100), target_id=51),
        ])
        self.assertEqual(len(result.actions), 1)
        self.assertEqual(result.actions[0].target_id, 51)

    def test_g_is_concurrent_with_normal_play(self):
        result = self.engine.decide([
            detection(TargetType.G_SYMBOL, None, (50, 200), target_id=60),
            detection(TargetType.NORMAL_BALLOON, BalloonColor.YELLOW, (300, 200), target_id=61),
        ])

        self.assertEqual(result.mode, GameMode.NORMAL_PLAY)
        self.assertTrue(self.engine.state.g_active)
        self.assertIsNotNone(result.g_action)
        self.assertTrue(result.g_action.is_g)
        self.assertEqual(result.g_action.target_type, TargetType.G_SYMBOL)
        self.assertEqual(len(result.actions), 1)
        self.assertEqual(result.actions[0].target_id, 61)

        # The G disappears; it is active state, not a sticky game mode.
        result = self.engine.decide([
            detection(TargetType.NORMAL_BALLOON, BalloonColor.BLUE, (400, 200), target_id=62),
        ])
        self.assertFalse(self.engine.state.g_active)
        self.assertIsNone(result.g_action)

    def test_g_is_concurrent_with_special_sequence(self):
        result = self.engine.decide([
            detection(TargetType.G_SYMBOL, None, (50, 200), target_id=70),
            detection(TargetType.SPECIAL_SEQUENCE_BALLOON, BalloonColor.BLUE, (500, 200), target_id=71),
            detection(TargetType.NORMAL_BALLOON, BalloonColor.BLUE, (700, 200), target_id=72),
        ])

        self.assertEqual(result.mode, GameMode.SPECIAL_SEQUENCE)
        self.assertIsNotNone(result.g_action)
        self.assertEqual(result.g_action.target_type, TargetType.G_SYMBOL)
        self.assertEqual(len(result.actions), 1)
        self.assertEqual(result.actions[0].target_id, 71)

    def test_multicolor_special_is_preserved_but_has_no_invented_priority(self):
        result = self.engine.decide([
            detection(TargetType.MULTICOLOR_SPECIAL, BalloonColor.BLUE, (100, 300), target_id=80),
            detection(TargetType.NORMAL_BALLOON, BalloonColor.RED, (300, 300), target_id=81),
        ])

        self.assertEqual(len(result.multicolor_specials), 1)
        self.assertEqual(
            result.multicolor_specials[0].target_type,
            TargetType.MULTICOLOR_SPECIAL,
        )
        self.assertEqual(len(result.actions), 1)
        self.assertEqual(result.actions[0].target_id, 81)


if __name__ == "__main__":
    unittest.main()
