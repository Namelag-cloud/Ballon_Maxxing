from __future__ import annotations

import unittest
import time
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import Settings, RulesConfig, GroupingConfig
from tracking.tracker import BalloonTrack
from decision.rules import SafetyRules
from decision.grouping import GroupingStrategy
from decision.selector import TargetSelector


class TestSafetyRules(unittest.TestCase):
    def _make_rules(self, **kwargs) -> SafetyRules:
        settings = Settings()
        for k, v in kwargs.items():
            setattr(settings.rules, k, v)
        return SafetyRules(settings)

    def test_reject_white(self):
        rules = self._make_rules(click_white=False)
        track = BalloonTrack(
            id=1, position=(100, 100), bbox=(50, 50, 100, 100),
            color="white", has_outline=False, confidence=0.9,
            first_seen=0.0, last_seen=0.0,
        )
        is_safe, reason = rules.is_safe(track)
        self.assertFalse(is_safe)
        self.assertEqual(reason, "white_balloon")

    def test_allow_white_when_configured(self):
        rules = self._make_rules(click_white=True)
        track = BalloonTrack(
            id=1, position=(100, 100), bbox=(50, 50, 100, 100),
            color="white", has_outline=False, confidence=0.9,
            first_seen=0.0, last_seen=0.0,
        )
        is_safe, reason = rules.is_safe(track)
        self.assertTrue(is_safe)

    def test_reject_too_small(self):
        rules = self._make_rules(min_size=500)
        track = BalloonTrack(
            id=1, position=(100, 100), bbox=(0, 0, 10, 10),
            color="red", has_outline=False, confidence=0.9,
            first_seen=0.0, last_seen=0.0,
        )
        is_safe, reason = rules.is_safe(track)
        self.assertFalse(is_safe)
        self.assertEqual(reason, "too_small")

    def test_reject_already_clicked(self):
        rules = self._make_rules()
        track = BalloonTrack(
            id=1, position=(100, 100), bbox=(0, 0, 100, 100),
            color="red", has_outline=False, confidence=0.9,
            first_seen=0.0, last_seen=0.0, clicked=True,
        )
        is_safe, reason = rules.is_safe(track)
        self.assertFalse(is_safe)
        self.assertEqual(reason, "already_clicked")

    def test_allow_normal(self):
        rules = self._make_rules()
        track = BalloonTrack(
            id=1, position=(100, 100), bbox=(50, 50, 100, 100),
            color="red", has_outline=False, confidence=0.9,
            first_seen=0.0, last_seen=0.0,
        )
        is_safe, reason = rules.is_safe(track)
        self.assertTrue(is_safe)


class TestGrouping(unittest.TestCase):
    def _make_grouping(self, **kwargs) -> GroupingStrategy:
        settings = Settings()
        for k, v in kwargs.items():
            setattr(settings.grouping, k, v)
        return GroupingStrategy(settings)

    def test_single_balloon(self):
        grouping = self._make_grouping(enabled=True, same_color_only=True)
        tracks = [
            BalloonTrack(id=1, position=(100, 100), bbox=(50, 50, 100, 100),
                         color="red", has_outline=False, confidence=0.9,
                         first_seen=0.0, last_seen=0.0),
        ]
        groups = grouping.group(tracks)
        self.assertEqual(len(groups), 1)
        self.assertEqual(groups[0].color, "red")
        self.assertEqual(len(groups[0].tracks), 1)

    def test_same_color_close(self):
        grouping = self._make_grouping(enabled=True, same_color_only=True, max_pair_distance=200)
        tracks = [
            BalloonTrack(id=1, position=(100, 100), bbox=(50, 50, 100, 100),
                         color="red", has_outline=False, confidence=0.9,
                         first_seen=0.0, last_seen=0.0),
            BalloonTrack(id=2, position=(200, 100), bbox=(150, 50, 100, 100),
                         color="red", has_outline=False, confidence=0.9,
                         first_seen=0.0, last_seen=0.0),
        ]
        groups = grouping.group(tracks)
        self.assertEqual(len(groups), 1)
        self.assertEqual(len(groups[0].tracks), 2)

    def test_different_colors(self):
        grouping = self._make_grouping(enabled=True, same_color_only=True, max_pair_distance=200)
        tracks = [
            BalloonTrack(id=1, position=(100, 100), bbox=(50, 50, 100, 100),
                         color="red", has_outline=False, confidence=0.9,
                         first_seen=0.0, last_seen=0.0),
            BalloonTrack(id=2, position=(150, 100), bbox=(100, 50, 100, 100),
                         color="blue", has_outline=False, confidence=0.9,
                         first_seen=0.0, last_seen=0.0),
        ]
        groups = grouping.group(tracks)
        self.assertEqual(len(groups), 2)

    def test_disabled_grouping(self):
        grouping = self._make_grouping(enabled=False)
        tracks = [
            BalloonTrack(id=1, position=(100, 100), bbox=(50, 50, 100, 100),
                         color="red", has_outline=False, confidence=0.9,
                         first_seen=0.0, last_seen=0.0),
            BalloonTrack(id=2, position=(200, 100), bbox=(150, 50, 100, 100),
                         color="red", has_outline=False, confidence=0.9,
                         first_seen=0.0, last_seen=0.0),
        ]
        groups = grouping.group(tracks)
        self.assertEqual(len(groups), 2)


class TestTargetSelector(unittest.TestCase):
    def test_select_safe_targets(self):
        settings = Settings()
        settings.rules.click_white = False
        settings.rules.min_size = 100
        settings.rules.max_size = 50000
        selector = TargetSelector(settings)

        now = time.time()
        tracks = {
            1: BalloonTrack(id=1, position=(100, 100), bbox=(50, 50, 100, 100),
                            color="red", has_outline=False, confidence=0.9,
                            first_seen=now, last_seen=now),
            2: BalloonTrack(id=2, position=(300, 300), bbox=(250, 250, 100, 100),
                            color="white", has_outline=False, confidence=0.9,
                            first_seen=now, last_seen=now),
        }

        actions = selector.select(tracks)
        self.assertEqual(len(actions), 1)
        self.assertEqual(actions[0].color, "red")

    def test_no_targets(self):
        settings = Settings()
        selector = TargetSelector(settings)
        actions = selector.select({})
        self.assertEqual(len(actions), 0)


if __name__ == "__main__":
    unittest.main()
