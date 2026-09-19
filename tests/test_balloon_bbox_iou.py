import unittest

from game_profiles.balloon_game import BalloonGameProfile


class TestBalloonBBoxIoU(unittest.TestCase):
    def test_identical_boxes_have_full_overlap(self):
        self.assertEqual(
            BalloonGameProfile._bbox_iou((10, 20, 30, 40), (10, 20, 30, 40)),
            1.0,
        )

    def test_disjoint_boxes_have_zero_overlap(self):
        self.assertEqual(
            BalloonGameProfile._bbox_iou((0, 0, 10, 10), (20, 20, 10, 10)),
            0.0,
        )

    def test_partial_overlap_matches_expected_iou(self):
        # Intersection: 5x5=25; union: 100+100-25=175.
        self.assertAlmostEqual(
            BalloonGameProfile._bbox_iou((0, 0, 10, 10), (5, 5, 10, 10)),
            25.0 / 175.0,
        )

    def test_degenerate_boxes_have_zero_overlap(self):
        self.assertEqual(
            BalloonGameProfile._bbox_iou((0, 0, 0, 10), (0, 0, 10, 10)),
            0.0,
        )


if __name__ == "__main__":
    unittest.main()
