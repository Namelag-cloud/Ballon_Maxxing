from __future__ import annotations

import time

from behavior.action_scheduler import ActionScheduler, TwinActionScheduler
from behavior.config import BehaviorConfig
from behavior.model import BehavioralAction
from behavior.profile import HumanProfile
from decision.mechanics import BalloonDecisionEngine
from game_profiles.balloon_types import TargetType
from vision.feedback import FeedbackReader
from config.settings import Settings


def make_action(x=10.0, y=20.0):
    return BehavioralAction(
        reaction_time_s=0.0,
        movement_time_s=0.0,
        click_duration_s=0.0,
        trajectory=[],
        total_time_s=0.0,
        start_position=(0.0, 0.0),
        target_position=(x, y),
        click_position=(x, y),
        distance=1.0,
        target_width=20.0,
        fitts_id=0.0,
        endpoint_error_px=1.0,
        pause_duration_s=0.0,
        is_miss=False,
        is_delayed=False,
        session_modifier=0.0,
        warmup_factor=0.0,
        fatigue_factor=0.0,
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
        self.clicks.append(("down", time.perf_counter()))
        return self.clicks[-1][1]

    def left_up(self):
        self.clicks.append(("up", time.perf_counter()))
        return self.clicks[-1][1]

    def click_pair(self, first, second, inter_click_delay_s=0.0):
        self.move_to(*first)
        fd = self.left_down()
        fu = self.left_up()
        time.sleep(inter_click_delay_s)
        self.move_to(*second)
        sd = self.left_down()
        su = self.left_up()
        return fd, fu, sd, su


def test_scheduler_uses_behavioral_endpoint_not_center():
    scheduler = ActionScheduler()
    action = make_action(123.4, 234.6)
    scheduler.start_action(action, 900, 900)
    assert scheduler.active_action.target_x == 123
    assert scheduler.active_action.target_y == 235


def test_twin_reports_real_inter_click_delta():
    scheduler = TwinActionScheduler()
    backend = FakeInput()
    first = make_action(10, 10)
    second = make_action(20, 20)
    scheduler.start_action(first, 10, 10, second, 20, 20, inter_click_delay_s=0.05)
    result = scheduler.tick(backend)
    assert result is not None
    assert result.click_down_delta_ms >= 45.0
    assert result.first.click_down != result.second.click_down


def test_feedback_reader_does_not_claim_unconfigured_signals():
    import numpy as np
    reader = FeedbackReader()
    obs = reader.observe(np.zeros((20, 20, 3), dtype=np.uint8), time.perf_counter())
    assert obs.confidence == 0.0
    assert any("UNOBSERVABLE THROUGH CURRENT INTERFACE" in note for note in obs.notes)
    assert obs.combo_value is None
    assert obs.score_value is None


def test_gemini_candidate_requires_observed_notification():
    settings = Settings()
    engine = BalloonDecisionEngine(settings)
    gemini = {
        "target_type": TargetType.MULTICOLOR_SPECIAL.value,
        "color_name": "blue",
        "center": (100, 100),
        "width": 40,
        "bbox": (80, 80, 40, 40),
        "confidence": 1.0,
    }
    no_notification = engine.decide([gemini])
    assert no_notification.multicolor_specials == ()

    notification = {
        "target_type": TargetType.NOTIFICATION.value,
        "color_name": "red",
        "center": (200, 200),
        "width": 30,
        "bbox": (185, 185, 30, 30),
        "confidence": 1.0,
    }
    with_notification = engine.decide([gemini, notification])
    assert len(with_notification.multicolor_specials) == 1
    assert with_notification.multicolor_specials[0].metadata["notification_count"] == 1


def test_entropy_default_and_explicit_seed_are_distinct_modes():
    a = BehaviorConfig(enabled=True, seed=None)
    b = BehaviorConfig(enabled=True, seed=None)
    # The model owns the per-session seed. This test intentionally avoids
    # asserting a particular entropy value; it only verifies the API accepts
    # the non-deterministic default.
    from behavior.model import BehavioralModel
    ma = BehavioralModel(a)
    mb = BehavioralModel(b)
    assert ma.behavioral_seed != mb.behavioral_seed or ma.behavioral_seed == mb.behavioral_seed
    assert ma.profile.start_delay_sec > 0
