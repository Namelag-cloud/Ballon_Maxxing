from __future__ import annotations

import unittest

from config.settings import Settings
from decision.mechanics import BalloonDecisionEngine
from game_profiles.balloon_types import BalloonColor, TargetType
from tracking.tracker import Tracker
from vision.detector import Detection
from vision.outline_detector import OutlineResult


def _detection(target_type, color, center, *, target_id=None):
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
        "metadata": {
            "semantic_confirmed": target_type == TargetType.MULTICOLOR_SPECIAL,
        },
    }


class Phase4SemanticTests(unittest.TestCase):
    def test_confirmed_gemini_is_first_action_when_normal_targets_exist(self):
        engine = BalloonDecisionEngine(Settings())
        result = engine.decide([
            _detection(TargetType.NORMAL_BALLOON, BalloonColor.RED, (100, 100), target_id=1),
            _detection(TargetType.NOTIFICATION, None, (300, 100), target_id=2),
            _detection(TargetType.NOTIFICATION, None, (350, 100), target_id=3),
            _detection(TargetType.MULTICOLOR_SPECIAL, None, (500, 100), target_id=4),
        ])

        self.assertEqual(result.actions[0].target_type, TargetType.MULTICOLOR_SPECIAL)
        self.assertEqual(result.actions[0].target_id, 4)

    def test_static_ui_suspect_is_not_selected_as_normal(self):
        engine = BalloonDecisionEngine(Settings())
        result = engine.decide([
            _detection(TargetType.NORMAL_BALLOON, BalloonColor.BLUE, (100, 100), target_id=1)
            | {"metadata": {
                "temporal": {
                    "age_frames": 20,
                    "stationary_frames": 18,
                    "mean_displacement_px": 0.2,
                    "max_bbox_delta_px": 0.0,
                    "static_ui_suspect": True,
                }
            }},
            _detection(TargetType.NORMAL_BALLOON, BalloonColor.RED, (300, 100), target_id=2),
        ])

        self.assertEqual([a.target_id for a in result.actions], [2])

    def test_tracker_emits_temporal_evidence_and_static_ui_signal(self):
        tracker = Tracker(Settings())
        det = Detection(
            bbox=(80, 80, 40, 40),
            center=(100, 100),
            area=1200,
            color_name="blue",
            color_confidence=0.95,
            outline=OutlineResult(),
            tracking_class=TargetType.NORMAL_BALLOON.value,
            metadata={"semantic_class": "normal"},
        )

        tracks = tracker.update([det])
        for _ in range(12):
            tracks = tracker.update([det])

        temporal = tracks[0].metadata["temporal"]
        self.assertGreaterEqual(temporal["age_frames"], 12)
        self.assertGreaterEqual(temporal["stationary_frames"], 10)
        self.assertTrue(temporal["static_ui_suspect"])


if __name__ == "__main__":
    unittest.main()
