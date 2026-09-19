import unittest

import cv2
import numpy as np

from game_profiles.balloon_game import BalloonGameProfile
from vision.ring_detector import RingDetector


class BalloonVisionProfilingTests(unittest.TestCase):
    def test_profile_fields_and_output_stability(self):
        profile = BalloonGameProfile()
        frame = np.zeros((180, 240, 3), dtype=np.uint8)

        first = profile.detect(frame)
        first_profile = profile.last_profile
        second = profile.detect(frame)
        second_profile = profile.last_profile

        self.assertEqual(first, second)

        required = {
            "total_ms",
            "hsv_ms",
            "blue_ms",
            "red_ms",
            "yellow_ms",
            "green_ms",
            "blue_mask_ms",
            "red_mask_ms",
            "yellow_mask_ms",
            "green_mask_ms",
            "blue_morphology_ms",
            "red_morphology_ms",
            "yellow_morphology_ms",
            "green_morphology_ms",
            "blue_components_ms",
            "red_components_ms",
            "yellow_components_ms",
            "green_components_ms",
            "blue_component_analysis_ms",
            "blue_component_mask_allocation_ms",
            "blue_component_mask_label_ms",
            "blue_component_classification_ms",
            "red_component_analysis_ms",
            "red_component_mask_allocation_ms",
            "red_component_mask_label_ms",
            "red_component_classification_ms",
            "yellow_component_analysis_ms",
            "yellow_component_mask_allocation_ms",
            "yellow_component_mask_label_ms",
            "yellow_component_classification_ms",
            "green_component_analysis_ms",
            "green_component_mask_allocation_ms",
            "green_component_mask_label_ms",
            "green_component_classification_ms",
            "multicolor_ms",
            "g_detector_ms",
            "ring_total_ms",
            "ring_hough_ms",
            "ring_grayscale_ms",
            "ring_blur_ms",
            "ring_canny_ms",
            "ring_scoring_ms",
            "classification_ms",
            "normal_candidate_count",
            "ring_call_count",
            "hough_call_count",
            "component_count",
            "candidate_counts",
            "candidate_geometry",
            "ring_circles_returned",
            "semantic",
        }
        self.assertTrue(required.issubset(first_profile))
        self.assertTrue(required.issubset(second_profile))
        self.assertEqual(first_profile["candidate_geometry"], second_profile["candidate_geometry"])

        timing_keys = required - {
            "candidate_counts",
            "normal_candidate_count",
            "ring_call_count",
            "hough_call_count",
            "component_count",
            "ring_circles_returned",
            "semantic",
            "candidate_geometry",
        }
        for key in timing_keys:
            self.assertGreaterEqual(first_profile[key], 0.0)

        for key in ("normal_candidate_count", "ring_call_count", "hough_call_count", "component_count", "ring_circles_returned"):
            self.assertIsInstance(first_profile[key], int)
            self.assertGreaterEqual(first_profile[key], 0)

        for color in ("blue", "red", "yellow", "green"):
            self.assertIsInstance(first_profile["candidate_counts"][color], int)
            self.assertGreaterEqual(first_profile["candidate_counts"][color], 0)

    def test_hollow_colored_ui_ring_is_not_a_balloon(self):
        profile = BalloonGameProfile()
        frame = np.full((180, 240, 3), 255, dtype=np.uint8)

        # A hollow UI glyph: saturated colored edge around an
        # enclosed background hole. It deliberately satisfies the size/shape
        # regime of the prior false-positive path.
        cv2.circle(frame, (120, 140), 31, (0, 0, 255), thickness=12)

        detections = profile.detect(frame)

        balloon_types = {"normal_balloon", "special_sequence_balloon", "twin_balloon"}
        self.assertFalse(
            any(d.get("target_type") in balloon_types for d in detections)
        )

    def test_non_balloon_aspect_ratios_are_rejected(self):
        profile = BalloonGameProfile()
        frame = np.full((240, 320, 3), 255, dtype=np.uint8)

        # Empirical false-positive shape: ~59x101 (aspect ~0.58).
        cv2.ellipse(frame, (80, 120), (29, 50), 0, 0, 360, (0, 0, 255), thickness=-1)
        # Another observed UI candidate was much wider than a balloon (~316x147).
        cv2.ellipse(frame, (240, 120), (100, 46), 0, 0, 360, (255, 0, 0), thickness=-1)

        detections = profile.detect(frame)

        balloon_types = {
            "normal_balloon",
            "special_sequence_balloon",
            "twin_balloon",
        }
        self.assertFalse(
            any(d.get("target_type") in balloon_types for d in detections)
        )
        self.assertFalse(
            any(
                candidate["aspect_ratio"] < profile._CORE_MIN_ASPECT_RATIO
                or candidate["aspect_ratio"] > profile._CORE_MAX_ASPECT_RATIO
                for candidate in profile.last_profile["candidate_geometry"]
            )
        )

    def test_boundary_radial_uniformity_accepts_balloon_like_circle(self):
        profile = BalloonGameProfile()
        frame = np.full((180, 240, 3), 255, dtype=np.uint8)
        cv2.circle(frame, (120, 80), 25, (255, 0, 0), thickness=-1)

        detections = profile.detect(frame)

        blue = [
            d for d in detections
            if d.get("target_type") == "normal_balloon"
            and d.get("color_name") == "blue"
        ]
        self.assertTrue(blue)
        candidate = next(
            g for g in profile.last_profile["candidate_geometry"]
            if g["color"] == "blue"
        )
        self.assertLessEqual(
            candidate["boundary_radial_cv"],
            profile._MAX_BOUNDARY_RADIAL_CV,
        )

    def test_boundary_radial_uniformity_rejects_square_like_ui_shape(self):
        profile = BalloonGameProfile()
        frame = np.full((180, 240, 3), 255, dtype=np.uint8)
        # A compact UI-like blue region can satisfy the existing bbox,
        # fill, circularity, and solidity gates while having an asymmetric
        # boundary that is not balloon-like.
        cv2.circle(frame, (120, 80), 25, (255, 0, 0), thickness=-1)
        cv2.rectangle(frame, (145, 73), (160, 87), (255, 0, 0), thickness=-1)

        contour = max(
            cv2.findContours(
                cv2.inRange(
                    cv2.cvtColor(frame, cv2.COLOR_BGR2HSV),
                    np.array([95, 80, 120], dtype=np.uint8),
                    np.array([135, 255, 255], dtype=np.uint8),
                ),
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_NONE,
            )[0],
            key=cv2.contourArea,
        )
        radial_cv = profile._boundary_radial_cv(contour)

        self.assertGreater(radial_cv, profile._MAX_BOUNDARY_RADIAL_CV)
        detections = profile.detect(frame)
        self.assertFalse(
            any(
                d.get("target_type") == "normal_balloon"
                and d.get("color_name") == "blue"
                for d in detections
            )
        )


    def test_boundary_contrast_is_recorded_for_balloon_candidates(self):
        profile = BalloonGameProfile()
        frame = np.full((180, 240, 3), 255, dtype=np.uint8)
        cv2.circle(frame, (120, 80), 25, (255, 0, 0), thickness=-1)

        detections = profile.detect(frame)
        self.assertTrue(any(d.get("target_type") == "normal_balloon" for d in detections))
        candidate = next(
            g for g in profile.last_profile["candidate_geometry"]
            if g["color"] == "blue"
        )
        self.assertGreaterEqual(
            candidate["boundary_contrast"],
            profile._MIN_BOUNDARY_CONTRAST,
        )

    def test_filled_colored_balloon_remains_detectable(self):
        profile = BalloonGameProfile()
        frame = np.full((180, 240, 3), 255, dtype=np.uint8)

        cv2 = __import__("cv2")
        cv2.circle(frame, (120, 80), 25, (0, 0, 255), thickness=-1)

        detections = profile.detect(frame)

        self.assertTrue(
            any(
                d.get("target_type") == "normal_balloon"
                and d.get("color_name") == "red"
                for d in detections
            )
        )
        geometry = profile.last_profile["candidate_geometry"]
        self.assertTrue(any(g["color"] == "red" for g in geometry))
        candidate = next(g for g in geometry if g["color"] == "red")
        self.assertEqual(candidate["width"], candidate["height"])
        self.assertGreater(candidate["circularity"], 0.45)

    def test_ring_profile_fields_and_output_stability(self):
        detector = RingDetector()
        frame = np.zeros((160, 200, 3), dtype=np.uint8)
        bbox = (80, 60, 20, 20)

        first = detector.detect(frame, bbox)
        first_profile = detector.last_profile
        second = detector.detect(frame, bbox)
        second_profile = detector.last_profile

        self.assertEqual(first, second)

        for key in (
            "total_ms",
            "crop_ms",
            "grayscale_ms",
            "blur_ms",
            "hough_ms",
            "canny_ms",
            "scoring_ms",
        ):
            self.assertIn(key, first_profile)
            self.assertGreaterEqual(first_profile[key], 0.0)

        self.assertEqual(first_profile["hough_called"], 1)
        self.assertIsInstance(first_profile["circles_returned"], int)
        self.assertGreaterEqual(first_profile["circles_returned"], 0)
        self.assertEqual(len(first_profile["crop_shape"]), 2)
        self.assertEqual(first_profile["crop_shape"], second_profile["crop_shape"])


if __name__ == "__main__":
    unittest.main()
