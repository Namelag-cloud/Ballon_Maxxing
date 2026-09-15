from __future__ import annotations

import unittest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import Settings, ColorThreshold
from vision.color_classifier import ColorClassifier


class TestColorClassifier(unittest.TestCase):
    def _make_classifier(self, colors: dict[str, tuple[list, list]]) -> ColorClassifier:
        settings = Settings()
        for name, (lower, upper) in colors.items():
            settings.colors[name] = ColorThreshold(lower_hsv=lower, upper_hsv=upper)
        return ColorClassifier(settings)

    def test_classify_red(self):
        classifier = self._make_classifier({
            "red": ([0, 120, 100], [10, 255, 255]),
            "green": ([35, 120, 100], [85, 255, 255]),
            "white": ([0, 0, 200], [180, 30, 255]),
        })

        hsv_frame = np.zeros((100, 100, 3), dtype=np.uint8)
        hsv_frame[20:80, 20:80] = [5, 200, 200]

        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[20:80, 20:80] = 255

        name, conf = classifier.classify_mask(hsv_frame, mask)
        self.assertEqual(name, "red")
        self.assertGreater(conf, 0.9)

    def test_classify_green(self):
        classifier = self._make_classifier({
            "red": ([0, 120, 100], [10, 255, 255]),
            "green": ([35, 120, 100], [85, 255, 255]),
            "white": ([0, 0, 200], [180, 30, 255]),
        })

        hsv_frame = np.zeros((100, 100, 3), dtype=np.uint8)
        hsv_frame[20:80, 20:80] = [60, 200, 200]

        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[20:80, 20:80] = 255

        name, conf = classifier.classify_mask(hsv_frame, mask)
        self.assertEqual(name, "green")
        self.assertGreater(conf, 0.9)

    def test_classify_white(self):
        classifier = self._make_classifier({
            "red": ([0, 120, 100], [10, 255, 255]),
            "white": ([0, 0, 200], [180, 30, 255]),
        })

        hsv_frame = np.zeros((100, 100, 3), dtype=np.uint8)
        hsv_frame[20:80, 20:80] = [0, 10, 240]

        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[20:80, 20:80] = 255

        name, conf = classifier.classify_mask(hsv_frame, mask)
        self.assertEqual(name, "white")
        self.assertGreater(conf, 0.9)

    def test_make_all_nonwhite_mask(self):
        settings = Settings()
        settings.colors["white"] = ColorThreshold(lower_hsv=[0, 0, 200], upper_hsv=[180, 30, 255])
        classifier = ColorClassifier(settings)

        hsv_frame = np.zeros((100, 100, 3), dtype=np.uint8)
        hsv_frame[0:50, :] = [0, 10, 240]
        hsv_frame[50:100, :] = [5, 200, 200]

        mask = classifier.make_all_nonwhite_mask(hsv_frame)
        white_count = np.sum(mask == 0)
        nonwhite_count = np.sum(mask == 255)
        self.assertEqual(white_count, 5000)
        self.assertEqual(nonwhite_count, 5000)

    def test_is_white(self):
        settings = Settings()
        settings.colors["white"] = ColorThreshold(lower_hsv=[0, 0, 200], upper_hsv=[180, 30, 255])
        classifier = ColorClassifier(settings)

        hsv_frame = np.zeros((100, 100, 3), dtype=np.uint8)
        hsv_frame[20:80, 20:80] = [0, 10, 240]

        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[20:80, 20:80] = 255

        self.assertTrue(classifier.is_white(hsv_frame, mask))

        hsv_frame2 = np.zeros((100, 100, 3), dtype=np.uint8)
        hsv_frame2[20:80, 20:80] = [5, 200, 200]
        self.assertFalse(classifier.is_white(hsv_frame2, mask))


if __name__ == "__main__":
    unittest.main()
