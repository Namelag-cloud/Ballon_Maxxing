from __future__ import annotations

import time
import unittest

from behavior.action_scheduler import ActionScheduler, TwinActionScheduler
from behavior.config import BehaviorConfig, ErrorConfig, HumanProfileConfig
from behavior.model import BehavioralAction, BehavioralModel
from decision.mechanics import BalloonDecisionEngine
from game_profiles.balloon_types import TargetType
from vision.feedback import FeedbackReader
from config.settings import Settings


def make_action(x=10.0, y=20.0):
    return BehavioralAction(
        reaction_time_s=0.0, movement_time_s=0.0, click_duration_s=0.0,
        trajectory=[], total_time_s=0.0, start_position=(0.0, 0.0),
        target_position=(x, y), click_position=(x, y), distance=1.0,
        target_width=20.0, fitts_id=0.0, endpoint_error_px=1.0,
        pause_duration_s=0.0, is_miss=False, is_delayed=False,
        session_modifier=0.0, warmup_factor=0.0, fatigue_factor=0.0,
        intended_reaction_ticks=10.0,
    )


class FakeInput:
    def __init__(self):
        self.moves = []
        self.clicks = []

    def move_to(self, x, y):
        self.moves.append((x, y))
        return time.perf_counter()

    def left_down(self):
        ts = time.perf_counter()
        self.clicks.append(("down", ts))
        return ts

    def left_up(self):
        ts = time.perf_counter()
        self.clicks.append(("up", ts))
        return ts

    def click_pair(self, first, second, inter_click_delay_s=0.0):
        self.move_to(*first)
        fd = self.left_down()
        fu = self.left_up()
        time.sleep(inter_click_delay_s)
        self.move_to(*second)
        sd = self.left_down()
        su = self.left_up()
        return fd, fu, sd, su


class TestBehavioralReplayValidation(unittest.TestCase):
    def test_scheduler_uses_behavioral_endpoint_not_center(self):
        scheduler = ActionScheduler()
        action = make_action(123.4, 234.6)
        scheduler.start_action(action, 900, 900)
        self.assertEqual(scheduler.active_action.target_x, 123)
        self.assertEqual(scheduler.active_action.target_y, 235)

    def test_twin_reports_real_inter_click_delta(self):
        scheduler = TwinActionScheduler()
        backend = FakeInput()
        scheduler.start_action(make_action(10, 10), 10, 10, make_action(20, 20), 20, 20, inter_click_delay_s=0.05)
        result = scheduler.tick(backend)
        self.assertIsNotNone(result)
        assert result is not None
        self.assertGreaterEqual(result.click_down_delta_ms, 45.0)
        self.assertNotEqual(result.first.click_down, result.second.click_down)

    def test_feedback_reader_does_not_claim_unconfigured_signals(self):
        import numpy as np
        reader = FeedbackReader()
        obs = reader.observe(np.zeros((20, 20, 3), dtype=np.uint8), time.perf_counter())
        self.assertEqual(obs.confidence, 0.0)
        self.assertTrue(any("UNOBSERVABLE THROUGH CURRENT INTERFACE" in note for note in obs.notes))
        self.assertIsNone(obs.combo_value)
        self.assertIsNone(obs.score_value)

    def test_gemini_candidate_requires_observed_notification(self):
        engine = BalloonDecisionEngine(Settings())
        gemini = {
            "target_type": TargetType.MULTICOLOR_SPECIAL.value, "color_name": "blue",
            "center": (100, 100), "width": 40, "bbox": (80, 80, 40, 40), "confidence": 1.0,
        }
        self.assertEqual(engine.decide([gemini]).multicolor_specials, ())
        notification = {
            "target_type": TargetType.NOTIFICATION.value, "color_name": "red",
            "center": (200, 200), "width": 30, "bbox": (185, 185, 30, 30), "confidence": 1.0,
        }
        result = engine.decide([gemini, notification])
        self.assertEqual(len(result.multicolor_specials), 1)
        self.assertEqual(result.multicolor_specials[0].metadata["notification_count"], 1)

    def test_entropy_default_creates_session_seed(self):
        a = BehavioralModel(BehaviorConfig(enabled=True, seed=None))
        b = BehavioralModel(BehaviorConfig(enabled=True, seed=None))
        self.assertNotEqual(a.behavioral_seed, b.behavioral_seed)

    def test_explicit_behavioral_seed_is_reproducible(self):
        cfg = BehaviorConfig(enabled=True, seed=777)
        a = BehavioralModel(cfg)
        b = BehavioralModel(cfg)
        self.assertEqual(a.behavioral_seed, b.behavioral_seed)
        self.assertEqual(a.profile.as_dict(), b.profile.as_dict())

    def test_miss_endpoint_is_outside_target_radius(self):
        cfg = BehaviorConfig(
            enabled=True,
            seed=777,
            errors=ErrorConfig(enabled=True, miss_probability=1.0),
            human_profile=HumanProfileConfig(miss_rate_min=1.0, miss_rate_max=1.0),
        )
        action = BehavioralModel(cfg).plan_action(0, 0, 100, 100, 40)
        self.assertTrue(action.is_miss)
        self.assertGreaterEqual(action.endpoint_error_px, 1.5 * 20.0)


if __name__ == "__main__":
    unittest.main()
