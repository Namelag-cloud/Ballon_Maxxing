from __future__ import annotations

import unittest

from game_profiles.balloon_types import (
    BalloonColor,
    BalloonGameState,
    GameMode,
    SpecialSequenceStep,
    TargetType,
    TargetDetection,
)


class TestBalloonContracts(unittest.TestCase):
    def test_target_types_cover_required_classes(self):
        self.assertEqual(TargetType.NORMAL_BALLOON.value, "normal_balloon")
        self.assertEqual(TargetType.TWIN_BALLOON.value, "twin_balloon")
        self.assertEqual(TargetType.SPECIAL_SEQUENCE_BALLOON.value, "special_sequence_balloon")
        self.assertEqual(TargetType.G_SYMBOL.value, "g_symbol")
        self.assertEqual(TargetType.MULTICOLOR_SPECIAL.value, "multicolor_special")
        self.assertEqual(TargetType.INVALID_WHITE.value, "invalid_white")

    def test_state_required_color(self):
        state = BalloonGameState(
            mode=GameMode.SPECIAL_SEQUENCE,
            special_step=SpecialSequenceStep.YELLOW,
        )
        self.assertEqual(state.required_special_color, BalloonColor.YELLOW)

    def test_detection_serializes_for_existing_profile_api(self):
        detection = TargetDetection(
            target_type=TargetType.NORMAL_BALLOON,
            color=BalloonColor.BLUE,
            center=(10, 20),
            bbox=(0, 5, 20, 30),
            area=400,
            confidence=0.95,
            width=20.0,
            height=30.0,
        )
        data = detection.as_dict()
        self.assertEqual(data["target_type"], "normal_balloon")
        self.assertEqual(data["color_name"], "blue")
        self.assertEqual(data["center"], (10, 20))
        self.assertIn("visual_features", data)


if __name__ == "__main__":
    unittest.main()
