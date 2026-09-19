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

    def test_persistent_normal_is_suppressed_after_two_click_attempts(self):
        engine = BalloonDecisionEngine(Settings())
        detection = _detection(TargetType.NORMAL_BALLOON, BalloonColor.BLUE, (100, 100), target_id=1)

        self.assertEqual(len(engine.decide([detection]).actions), 1)
        detection["metadata"]["click_attempts"] = 1
        self.assertEqual(len(engine.decide([detection]).actions), 1)
        detection["metadata"]["click_attempts"] = 2
        self.assertEqual(engine.decide([detection]).actions, ())

    def test_twin_is_not_suppressed_by_click_attempt_count(self):
        engine = BalloonDecisionEngine(Settings())
        first = _detection(TargetType.TWIN_BALLOON, BalloonColor.GREEN, (100, 100), target_id=1)
        second = _detection(TargetType.TWIN_BALLOON, BalloonColor.GREEN, (180, 100), target_id=2)
        for d in (first, second):
            d["metadata"].update({"semantic_confirmed": True, "semantic_pair_id": "pair-1", "click_attempts": 5})

        result = engine.decide([first, second])
        self.assertEqual(len(result.actions), 1)
        self.assertEqual(type(result.actions[0]).__name__, "TwinAction")

    def test_tracker_click_attempts_only_apply_to_normal_balloons(self):
        tracker = Tracker(Settings())
        normal = Detection(
            bbox=(80, 80, 40, 40), center=(100, 100), area=1200,
            color_name="blue", color_confidence=0.95,
            outline=OutlineResult(has_outline=False, white_ratio=0.0, contrast=0.0),
            tracking_class=TargetType.NORMAL_BALLOON.value,
        )
        twin = Detection(
            bbox=(180, 80, 40, 40), center=(200, 100), area=1200,
            color_name="green", color_confidence=0.95,
            outline=OutlineResult(has_outline=False, white_ratio=0.0, contrast=0.0),
            tracking_class=TargetType.TWIN_BALLOON.value,
        )
        tracks = tracker.update([normal, twin])
        self.assertTrue(tracker.mark_clicked(0))
        self.assertTrue(tracker.mark_clicked(0))
        self.assertEqual(tracks[0].click_attempts, 2)
        self.assertFalse(tracker.mark_clicked(1))
        self.assertEqual(tracks[1].click_attempts, 0)

    
    def test_spatial_suppression_survives_tracker_churn(self):
        tracker = Tracker(Settings())
        det = Detection(
            bbox=(80, 80, 40, 40), center=(100, 100), area=1200,
            color_name="blue", color_confidence=0.95,
            outline=OutlineResult(has_outline=False, white_ratio=0.0, contrast=0.0),
            tracking_class=TargetType.NORMAL_BALLOON.value,
        )
        tracker.update([det])
        tracker.mark_clicked(0)
        tracker.mark_clicked(0)

        # Lose the old track long enough to force a new track id.
        for _ in range(6):
            tracker.update([])
        tracks = tracker.update([det])

        self.assertEqual(tracks, {})

    def test_super_g_sequence_is_exclusive_and_ordered(self):
        engine = BalloonDecisionEngine(Settings())
        specials = []
        for idx, color in enumerate((
            BalloonColor.BLUE,
            BalloonColor.RED,
            BalloonColor.YELLOW,
            BalloonColor.GREEN,
        ), start=1):
            item = _detection(TargetType.SPECIAL_SEQUENCE_BALLOON, color, (idx * 100, 100), target_id=idx)
            item["metadata"]["semantic_confirmed"] = True
            specials.append(item)

        first = engine.decide(specials)
        self.assertEqual(engine.state.mode.value, "special_sequence")
        self.assertEqual([a.color for a in first.actions], [BalloonColor.BLUE])
        self.assertIsNone(first.g_action)

        engine.complete_action(first.actions[0])
        second = engine.decide(specials)
        self.assertEqual([a.color for a in second.actions], [BalloonColor.RED])

        engine.complete_action(second.actions[0])
        third = engine.decide(specials)
        self.assertEqual([a.color for a in third.actions], [BalloonColor.YELLOW])

        engine.complete_action(third.actions[0])
        fourth = engine.decide(specials)
        self.assertEqual([a.color for a in fourth.actions], [BalloonColor.GREEN])

        engine.complete_action(fourth.actions[0])
        self.assertEqual(engine.state.mode.value, "normal_play")

    def test_tracker_emits_temporal_evidence_and_static_ui_signal(self):
        tracker = Tracker(Settings())
        det = Detection(
            bbox=(80, 80, 40, 40),
            center=(100, 100),
            area=1200,
            color_name="blue",
            color_confidence=0.95,
            outline=OutlineResult(has_outline=False, white_ratio=0.0, contrast=0.0),
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
