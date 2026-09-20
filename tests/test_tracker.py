from __future__ import annotations

import unittest
import time
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import Settings, TrackingConfig
from tracking.tracker import Tracker, BalloonTrack
from vision.detector import Detection
from vision.outline_detector import OutlineResult


class TestTracker(unittest.TestCase):
    def _make_tracker(self, **kwargs) -> Tracker:
        settings = Settings()
        for k, v in kwargs.items():
            setattr(settings.tracking, k, v)
        return Tracker(settings)

    def _make_detection(self, cx: int, cy: int, color: str = "red") -> Detection:
        return Detection(
            bbox=(cx - 25, cy - 25, 50, 50),
            center=(cx, cy),
            area=2500,
            color_name=color,
            color_confidence=0.95,
            outline=OutlineResult(has_outline=False, white_ratio=0.0, contrast=0.0),
        )

    def test_returns_dict(self):
        tracker = self._make_tracker()
        det = self._make_detection(100, 100)
        tracks = tracker.update([det])
        self.assertIsInstance(tracks, dict)

    def test_creates_new_track(self):
        tracker = self._make_tracker()
        det = self._make_detection(100, 100)
        tracks = tracker.update([det])
        self.assertEqual(len(tracks), 1)
        track = next(iter(tracks.values()))
        self.assertEqual(track.position, (100, 100))
        self.assertEqual(track.color, "red")

    def test_stores_first_frame_capture_timestamp(self):
        tracker = self._make_tracker(max_distance=50)
        det = self._make_detection(100, 100)
        tracks = tracker.update([det], capture_timestamp=123.456)
        track = next(iter(tracks.values()))
        self.assertEqual(track.capture_timestamp, 123.456)

        tracks = tracker.update([self._make_detection(105, 105)], capture_timestamp=200.0)
        track = next(iter(tracks.values()))
        self.assertEqual(track.capture_timestamp, 123.456)

    def test_updates_existing_track(self):
        tracker = self._make_tracker(max_distance=50)
        det1 = self._make_detection(100, 100)
        tracker.update([det1])

        det2 = self._make_detection(105, 105)
        tracks = tracker.update([det2])
        self.assertEqual(len(tracks), 1)
        track = next(iter(tracks.values()))
        self.assertEqual(track.position, (105, 105))

    def test_creates_second_track(self):
        tracker = self._make_tracker(max_distance=50)
        det1 = self._make_detection(100, 100)
        det2 = self._make_detection(400, 400)
        tracks = tracker.update([det1, det2])
        self.assertEqual(len(tracks), 2)

    def test_removes_stale_track(self):
        tracker = self._make_tracker(max_distance=50, max_frames_missing=2)
        det = self._make_detection(100, 100)
        tracker.update([det])
        self.assertEqual(len(tracker.tracks), 1)

        tracker.update([])
        self.assertEqual(len(tracker.tracks), 1)

        tracker.update([])
        self.assertEqual(len(tracker.tracks), 0)

    def test_track_id_stability(self):
        tracker = self._make_tracker(max_distance=50)
        det = self._make_detection(100, 100)
        tracks1 = tracker.update([det])
        id1 = next(iter(tracks1.keys()))

        det2 = self._make_detection(108, 108)
        tracks2 = tracker.update([det2])
        id2 = next(iter(tracks2.keys()))

        self.assertEqual(id1, id2)

    def test_dict_keys_are_track_ids(self):
        tracker = self._make_tracker(max_distance=50)
        det1 = self._make_detection(100, 100)
        det2 = self._make_detection(400, 400)
        tracks = tracker.update([det1, det2])

        for track_id, track in tracks.items():
            self.assertEqual(track_id, track.id)


if __name__ == "__main__":
    unittest.main()
