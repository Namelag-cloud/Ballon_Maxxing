from __future__ import annotations

import unittest

from config.settings import Settings
from tracking.tracker import BalloonTrack
from decision.grouping import GroupingStrategy


class TestGrouping(unittest.TestCase):
    def _make_grouping(self, **kwargs) -> GroupingStrategy:
        settings = Settings()
        for key, value in kwargs.items():
            setattr(settings.grouping, key, value)
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


if __name__ == "__main__":
    unittest.main()
