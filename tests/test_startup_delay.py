import sys
import os
import time
import argparse
import ctypes

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import parse_args


class TestStartupDelayCLI:
    def test_default_delay(self):
        args = parse_args(["--mode", "live", "--fullscreen"])
        assert args.startup_delay == 3.0

    def test_custom_delay(self):
        args = parse_args(["--mode", "live", "--fullscreen", "--startup-delay", "5.0"])
        assert args.startup_delay == 5.0

    def test_zero_delay(self):
        args = parse_args(["--mode", "live", "--fullscreen", "--startup-delay", "0"])
        assert args.startup_delay == 0.0

    def test_delay_not_applied_dryrun(self):
        args = parse_args(["--mode", "dry-run", "--fullscreen", "--startup-delay", "10.0"])
        assert args.startup_delay == 10.0

    def test_delay_not_applied_benchmark(self):
        args = parse_args(["--mode", "benchmark", "--fullscreen", "--startup-delay", "10.0"])
        assert args.startup_delay == 10.0


class TestStartupDelayBehavior:
    def test_zero_delay_starts_immediately(self):
        from config.settings import Settings, RunMode, load_settings
        from main import Application

        settings = load_settings()
        settings.mode = RunMode.DRY_RUN
        settings.capture.region.left = 0
        settings.capture.region.top = 0
        settings.capture.region.width = 1920
        settings.capture.region.height = 1080

        app = Application(settings, startup_delay=0.0)
        assert app._startup_delay == 0.0

    def test_positive_delay_stored(self):
        from config.settings import Settings, RunMode, load_settings
        from main import Application

        settings = load_settings()
        settings.mode = RunMode.DRY_RUN
        settings.capture.region.left = 0
        settings.capture.region.top = 0
        settings.capture.region.width = 1920
        settings.capture.region.height = 1080

        app = Application(settings, startup_delay=5.0)
        assert app._startup_delay == 5.0

    def test_input_disabled_during_countdown(self):
        from config.settings import load_settings, RunMode
        from main import Application
        from input.sendinput import SendInputBackend
        import unittest.mock as mock

        settings = load_settings()
        settings.mode = RunMode.LIVE
        settings.input.enabled = True
        settings.capture.region.left = 0
        settings.capture.region.top = 0
        settings.capture.region.width = 1920
        settings.capture.region.height = 1080

        app = Application(settings, startup_delay=0.1)
        app._input_backend = SendInputBackend()

        app._input_enabled = True
        original_click = app._input_backend.click_screen
        click_called = []

        def spy_click(sx, sy):
            click_called.append((sx, sy))
            return original_click(sx, sy)

        app._input_backend.click_screen = spy_click

        t_start = time.monotonic()
        delay_end = t_start + app._startup_delay
        while time.monotonic() < delay_end:
            time.sleep(0.01)

        assert len(click_called) == 0
