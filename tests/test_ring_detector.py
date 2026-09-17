from __future__ import annotations

import unittest

import cv2
import numpy as np

from vision.ring_detector import RingDetector


class TestRingDetector(unittest.TestCase):
    def _frame(self, *, ring: bool) -> np.ndarray:
        frame = np.full((500, 500, 3), 30, dtype=np.uint8)
        cv2.circle(frame, (250, 250), 60, (255, 80, 20), -1)
        if ring:
            cv2.circle(frame, (250, 250), 115, (255, 80, 20), 3)
        return frame

    def test_detects_large_ring_relative_to_balloon(self):
        detector = RingDetector(min_confidence=0.20)
        result = detector.detect(self._frame(ring=True), (190, 190, 120, 120))
        self.assertTrue(result.detected)
        self.assertGreater(result.radius_ratio, 1.4)
        self.assertLess(result.radius_ratio, 2.2)
        self.assertGreater(result.confidence, 0.20)

    def test_does_not_report_ring_for_plain_balloon(self):
        detector = RingDetector(min_confidence=0.20)
        result = detector.detect(self._frame(ring=False), (190, 190, 120, 120))
        self.assertFalse(result.detected)


if __name__ == "__main__":
    unittest.main()
