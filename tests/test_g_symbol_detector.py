from __future__ import annotations

import unittest
from unittest.mock import patch

import cv2

from config.settings import load_settings
from game_profiles.balloon_types import TargetType
from vision.color_classifier import ColorClassifier
from vision.g_symbol_detector import GSymbolDetector
from tests.fixtures.g_symbol_frames import g_symbol_frame, multicolor_star_frame


class TestGSymbolDetector(unittest.TestCase):
    def setUp(self) -> None:
        settings = load_settings()
        self.detector = GSymbolDetector(ColorClassifier(settings))

    def test_detects_four_color_g_geometry(self):
        hsv = cv2.cvtColor(g_symbol_frame(), cv2.COLOR_BGR2HSV)
        detections = self.detector.detect(hsv)
        self.assertEqual(len(detections), 1)
        detection = detections[0]
        self.assertEqual(detection.target_type, TargetType.G_SYMBOL)
        self.assertGreaterEqual(detection.confidence, 0.66)
        self.assertGreater(detection.width, detection.height * 0.5)
        self.assertLess(detection.width, detection.height * 1.1)
        self.assertGreater(detection.center[0], detection.bbox[0])

    def test_detects_g_when_yellow_is_missing_during_animation(self):
        hsv = cv2.cvtColor(
            g_symbol_frame(include_yellow=False),
            cv2.COLOR_BGR2HSV,
        )
        detections = self.detector.detect(hsv)
        self.assertEqual(len(detections), 1)
        self.assertEqual(detections[0].target_type, TargetType.G_SYMBOL)

    def test_counter_overlay_does_not_change_g_detection(self):
        plain = self.detector.detect(
            cv2.cvtColor(g_symbol_frame(), cv2.COLOR_BGR2HSV)
        )
        with_counter = self.detector.detect(
            cv2.cvtColor(g_symbol_frame(counter_overlay=True), cv2.COLOR_BGR2HSV)
        )
        self.assertEqual(len(plain), 1)
        self.assertEqual(len(with_counter), 1)
        self.assertEqual(
            with_counter[0].target_type,
            TargetType.G_SYMBOL,
        )
        self.assertEqual(with_counter[0].center, plain[0].center)

    def test_detects_left_edge_clipped_g(self):
        hsv = cv2.cvtColor(
            g_symbol_frame(left_clipped=True),
            cv2.COLOR_BGR2HSV,
        )
        detections = self.detector.detect(hsv)
        self.assertEqual(len(detections), 1)
        self.assertEqual(detections[0].target_type, TargetType.G_SYMBOL)
        self.assertEqual(detections[0].bbox[0], 0)

    def test_component_hints_bypass_rebuilding_color_masks(self):
        hsv = cv2.cvtColor(g_symbol_frame(), cv2.COLOR_BGR2HSV)
        hints = {
            color.value: [
                (component.bbox, component.area, component.center)
                for component in self.detector._components_for_color(hsv, color)
            ]
            for color in self.detector._COLORS
        }

        with patch.object(
            self.detector._classifier,
            "make_color_mask",
            side_effect=AssertionError("color masks should be reused"),
        ):
            detections = self.detector.detect(hsv, component_hints=hints)

        self.assertEqual(len(detections), 1)
        self.assertEqual(detections[0].target_type, TargetType.G_SYMBOL)

    def test_multicolor_star_is_not_g(self):
        detections = self.detector.detect(
            cv2.cvtColor(multicolor_star_frame(), cv2.COLOR_BGR2HSV)
        )
        self.assertEqual(
            [d for d in detections if d.target_type == TargetType.G_SYMBOL],
            [],
        )


if __name__ == "__main__":
    unittest.main()
