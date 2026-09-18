from __future__ import annotations

import time
import unittest

from behavior.action_scheduler import ActionScheduler, TwinActionScheduler
from behavior.config import BehaviorConfig, ErrorConfig, HumanProfileConfig
from behavior.model import BehavioralAction, BehavioralModel
from decision.mechanics import BalloonDecisionEngine
from game_profiles.balloon_types import TargetType
from vision.feedback import FeedbackReader
from telemetry.outcome import TimingCalibration
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

    def test_reaction_timing_keeps_perception_in_model_interval(self):
        cfg = BehaviorConfig(enabled=True, seed=123)
        action = BehavioralModel(cfg).plan_action(0, 0, 100, 100, 40)
        self.assertAlmostEqual(action.reaction_time_s, action.intended_reaction_ticks / 60.0, places=9)
        self.assertGreater(action.perception_ticks, 0.0)

    def test_click_tick_modulo_is_modulo_three_and_skips_missing(self):
        calibration = TimingCalibration()
        calibration.record(intended_ticks=10, observed_ticks=8.0, wall_clock_error_ms=1.0)
        calibration.record(intended_ticks=10, observed_ticks=None, wall_clock_error_ms=2.0)
        self.assertEqual(calibration.click_tick_modulo, [2])
        self.assertEqual(calibration.summary()["click_tick_modulo"]["count"], 1)

    def test_fat_finger_event_scales_endpoint_noise(self):
        from behavior.config import EndpointErrorConfig
        profile = HumanProfileConfig(spatial_sigma_pct_min=0.1, spatial_sigma_pct_max=0.1, fat_finger_probability=1.0, fat_finger_multiplier=2.0)
        cfg = BehaviorConfig(enabled=True, seed=9, endpoint_error=EndpointErrorConfig(enabled=False), human_profile=profile)
        fat = BehavioralModel(cfg).plan_action(0, 0, 100, 100, 40)
        normal_profile = HumanProfileConfig(spatial_sigma_pct_min=0.1, spatial_sigma_pct_max=0.1, fat_finger_probability=0.0, fat_finger_multiplier=2.0)
        normal = BehavioralModel(BehaviorConfig(enabled=True, seed=9, endpoint_error=EndpointErrorConfig(enabled=False), human_profile=normal_profile)).plan_action(0, 0, 100, 100, 40)
        self.assertAlmostEqual(fat.endpoint_error_px, normal.endpoint_error_px * 2.0, places=6)

    def test_duo_y_variance_multiplier(self):
        from behavior.config import EndpointErrorConfig
        base = HumanProfileConfig(spatial_sigma_pct_min=0.1, spatial_sigma_pct_max=0.1, fat_finger_probability=0.0, duo_y_variance_multiplier=1.0)
        duo = HumanProfileConfig(spatial_sigma_pct_min=0.1, spatial_sigma_pct_max=0.1, fat_finger_probability=0.0, duo_y_variance_multiplier=2.0)
        a = BehavioralModel(BehaviorConfig(enabled=True, seed=11, endpoint_error=EndpointErrorConfig(enabled=False), human_profile=base)).plan_action(0,0,100,100,40,target_metadata={"is_twin":False})
        b = BehavioralModel(BehaviorConfig(enabled=True, seed=11, endpoint_error=EndpointErrorConfig(enabled=False), human_profile=duo)).plan_action(0,0,100,100,40,target_metadata={"is_twin":True})
        self.assertAlmostEqual(b.click_position[0] - 100.0, a.click_position[0] - 100.0, places=6)
        self.assertAlmostEqual(b.click_position[1] - 100.0, 2.0 * (a.click_position[1] - 100.0), places=6)

    def test_early_exit_probability_is_sampled_once(self):
        profile = HumanProfileConfig(early_exit_prob_min=1.0, early_exit_prob_max=1.0, early_exit_ratio_min=0.5, early_exit_ratio_max=0.5)
        model = BehavioralModel(BehaviorConfig(enabled=True, seed=4, human_profile=profile))
        model.start_session()
        self.assertFalse(model.should_early_exit(10.0))
        model._session_start -= 6.0
        self.assertTrue(model.should_early_exit(10.0))

    def test_start_delay_and_final_activity_cutoff_are_enforced(self):
        profile = HumanProfileConfig(start_delay_min_s=2.0, start_delay_max_s=2.0, final_activity_cutoff_s=1.5)
        model = BehavioralModel(BehaviorConfig(enabled=True, seed=5, human_profile=profile))
        model.start_session()
        self.assertFalse(model.actions_ready())
        model._session_start -= 2.1
        self.assertTrue(model.actions_ready())
        model._session_start -= 6.0
        self.assertFalse(model.should_start_new_action(10.0))

    def test_profile_failure_and_gemini_probabilities_are_used(self):
        profile = HumanProfileConfig(
            super_g_fail_rate_min=1.0, super_g_fail_rate_max=1.0,
            balloon_g_abandon_rate_min=1.0, balloon_g_abandon_rate_max=1.0,
            gemini_without_notification_probability=1.0,
        )
        model = BehavioralModel(BehaviorConfig(enabled=True, seed=6, human_profile=profile))
        self.assertTrue(model.should_fail_super_g())
        self.assertTrue(model.should_abandon_balloon_g())
        self.assertTrue(model.should_select_gemini_without_notification())

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
        self.assertIsNone(obs.pop_detected)

    def test_gemini_candidate_requires_observed_notification(self):
        engine = BalloonDecisionEngine(Settings())
        gemini = {
            "target_type": TargetType.MULTICOLOR_SPECIAL.value, "color_name": "blue",
            "center": (100, 100), "width": 40, "bbox": (80, 80, 40, 40), "confidence": 1.0,
        }
        no_notification = engine.decide([gemini])
        self.assertEqual(len(no_notification.multicolor_specials), 1)
        self.assertEqual(no_notification.actions, ())
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
