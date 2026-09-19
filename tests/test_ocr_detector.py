from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np

from vision.ocr_detector import OCRDetector, OCRSnapshot, OCRWord


class OCRDetectorTests(unittest.TestCase):
    def _snapshot(self, *words: OCRWord, age: int = 1) -> OCRSnapshot:
        return OCRSnapshot(tuple(words), completed_frame_id=10, age_frames=age)

    def test_game_roi_crops_only_central_viewport(self):
        detector = OCRDetector()
        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        roi, origin_x, origin_y = detector._crop_game_roi(frame)

        self.assertEqual(origin_x, 540)
        self.assertEqual(origin_y, 75)
        self.assertEqual(roi.shape, (815, 575, 3))

    def test_ocr_word_coordinates_are_translated_back_to_screen_space(self):
        detector = OCRDetector()
        frame = np.zeros((20, 30, 3), dtype=np.uint8)
        tesseract_data = {
            "text": ["8"],
            "conf": ["96.0"],
            "left": [7],
            "top": [9],
            "width": [12],
            "height": [14],
        }

        with patch(
            "vision.ocr_detector.pytesseract.image_to_data",
            return_value=tesseract_data,
        ):
            words = detector._run_ocr(frame, origin=(540, 75))

        self.assertEqual(len(words), 1)
        self.assertEqual(words[0].bbox, (547, 84, 12, 14))

    def test_numeric_ocr_rejects_colored_score_component(self):
        detector = OCRDetector()
        result = detector.classify_bbox(
            (970, 245, 40, 105),
            snapshot=self._snapshot(OCRWord("8", 96.0, (980, 250, 28, 92))),
        )
        self.assertTrue(result["ocr_numeric"])
        self.assertFalse(result["ocr_twin_text"])

    def test_twin_phrase_is_semantic_evidence(self):
        detector = OCRDetector()
        result = detector.classify_bbox(
            (82, 96, 105, 38),
            snapshot=self._snapshot(
                OCRWord("pop", 91.0, (90, 110, 28, 10)),
                OCRWord("together", 94.0, (118, 110, 60, 10)),
            ),
        )
        self.assertTrue(result["ocr_twin_text"])
        self.assertFalse(result["ocr_numeric"])

    def test_stale_ocr_is_not_used_as_hard_veto(self):
        detector = OCRDetector()
        result = detector.classify_bbox(
            (970, 245, 40, 105),
            snapshot=self._snapshot(
                OCRWord("8", 99.0, (980, 250, 28, 92)),
                age=25,
            ),
        )
        self.assertFalse(result["ocr_numeric"])

    def test_unrelated_ocr_does_not_reject_balloon(self):
        detector = OCRDetector()
        result = detector.classify_bbox(
            (500, 500, 30, 30),
            snapshot=self._snapshot(OCRWord("score", 95.0, (100, 100, 40, 12))),
        )
        self.assertEqual(result["ocr_text"], [])
        self.assertFalse(result["ocr_ui_text"])


if __name__ == "__main__":
    unittest.main()
