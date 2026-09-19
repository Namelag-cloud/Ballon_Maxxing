from __future__ import annotations

import unittest

from vision.ocr_detector import OCRDetector, OCRSnapshot, OCRWord


class OCRDetectorTests(unittest.TestCase):
    def _snapshot(self, *words: OCRWord, age: int = 1) -> OCRSnapshot:
        return OCRSnapshot(tuple(words), completed_frame_id=10, age_frames=age)

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
