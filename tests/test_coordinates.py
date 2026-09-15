from __future__ import annotations

import unittest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from input.coordinates import (
    frame_to_screen,
    screen_to_sendinput,
    frame_to_sendinput,
)


class TestFrameToScreen(unittest.TestCase):
    def test_origin_capture(self):
        sx, sy = frame_to_screen(100, 200, 0, 0)
        self.assertEqual(sx, 100)
        self.assertEqual(sy, 200)

    def test_offset_capture(self):
        sx, sy = frame_to_screen(100, 200, 500, 300)
        self.assertEqual(sx, 600)
        self.assertEqual(sy, 500)

    def test_zero_frame_point(self):
        sx, sy = frame_to_screen(0, 0, 100, 200)
        self.assertEqual(sx, 100)
        self.assertEqual(sy, 200)

    def test_fullscreen_1920_1080(self):
        sx, sy = frame_to_screen(0, 0, 0, 0)
        self.assertEqual((sx, sy), (0, 0))

        sx, sy = frame_to_screen(1919, 1079, 0, 0)
        self.assertEqual((sx, sy), (1919, 1079))

        sx, sy = frame_to_screen(960, 540, 0, 0)
        self.assertEqual((sx, sy), (960, 540))

    def test_nonzero_region_800x600(self):
        region_left, region_top = 200, 100
        points = [
            (0, 0, 200, 100),
            (400, 300, 600, 400),
            (799, 599, 999, 699),
        ]
        for fx, fy, ex, ey in points:
            sx, sy = frame_to_screen(fx, fy, region_left, region_top)
            self.assertEqual((sx, sy), (ex, ey), f"frame=({fx},{fy})")


class TestScreenToSendinput(unittest.TestCase):
    def test_origin_maps_to_zero(self):
        nx, ny = screen_to_sendinput(0, 0, 0, 0, 1920, 1080)
        self.assertEqual(nx, 0)
        self.assertEqual(ny, 0)

    def test_bottom_right_maps_to_65535(self):
        nx, ny = screen_to_sendinput(1919, 1079, 0, 0, 1920, 1080)
        self.assertEqual(nx, 65535)
        self.assertEqual(ny, 65535)

    def test_center_1920_1080(self):
        nx, ny = screen_to_sendinput(960, 540, 0, 0, 1920, 1080)
        self.assertAlmostEqual(nx, 32767, delta=50)
        self.assertAlmostEqual(ny, 32767, delta=50)

    def test_negative_virtual_origin(self):
        nx, ny = screen_to_sendinput(-1920, 0, -1920, 0, 3840, 1080)
        self.assertEqual(nx, 0)
        self.assertEqual(ny, 0)

        nx, ny = screen_to_sendinput(0, 540, -1920, 0, 3840, 1080)
        self.assertAlmostEqual(nx, 32767, delta=50)
        self.assertAlmostEqual(ny, 32767, delta=50)

        nx, ny = screen_to_sendinput(1919, 1079, -1920, 0, 3840, 1080)
        self.assertEqual(nx, 65535)
        self.assertEqual(ny, 65535)

    def test_clamping_left(self):
        nx, ny = screen_to_sendinput(-100, 0, 0, 0, 1920, 1080)
        self.assertEqual(nx, 0)

    def test_clamping_right(self):
        nx, ny = screen_to_sendinput(2000, 0, 0, 0, 1920, 1080)
        self.assertEqual(nx, 65535)

    def test_clamping_top(self):
        nx, ny = screen_to_sendinput(0, -100, 0, 0, 1920, 1080)
        self.assertEqual(ny, 0)

    def test_clamping_bottom(self):
        nx, ny = screen_to_sendinput(0, 2000, 0, 0, 1920, 1080)
        self.assertEqual(ny, 65535)

    def test_round_trip(self):
        for sx, sy in [(0, 0), (100, 200), (960, 540), (1919, 1079)]:
            nx, ny = screen_to_sendinput(sx, sy, 0, 0, 1920, 1080)
            inv_x = int(nx * 1920 / 65535)
            inv_y = int(ny * 1080 / 65535)
            self.assertAlmostEqual(inv_x, sx, delta=1, msg=f"screen=({sx},{sy})")
            self.assertAlmostEqual(inv_y, sy, delta=1, msg=f"screen=({sx},{sy})")


class TestFrameToSendinput(unittest.TestCase):
    def test_fullscreen_origin(self):
        nx, ny = frame_to_sendinput(0, 0, 0, 0, 0, 0, 1920, 1080)
        self.assertEqual(nx, 0)
        self.assertEqual(ny, 0)

    def test_fullscreen_bottom_right(self):
        nx, ny = frame_to_sendinput(1919, 1079, 0, 0, 0, 0, 1920, 1080)
        self.assertEqual(nx, 65535)
        self.assertEqual(ny, 65535)

    def test_offset_region(self):
        nx, ny = frame_to_sendinput(100, 200, 500, 300, 0, 0, 1920, 1080)
        sx, sy = frame_to_screen(100, 200, 500, 300)
        self.assertEqual((sx, sy), (600, 500))
        expected_nx, expected_ny = screen_to_sendinput(600, 500, 0, 0, 1920, 1080)
        self.assertEqual(nx, expected_nx)
        self.assertEqual(ny, expected_ny)

    def test_negative_virtual_with_offset(self):
        nx, ny = frame_to_sendinput(0, 0, -1920, 0, -1920, 0, 3840, 1080)
        self.assertEqual(nx, 0)
        self.assertEqual(ny, 0)

    def test_clamps(self):
        nx, ny = frame_to_sendinput(9999, 9999, 0, 0, 0, 0, 1920, 1080)
        self.assertEqual(nx, 65535)
        self.assertEqual(ny, 65535)


if __name__ == "__main__":
    unittest.main()
