from __future__ import annotations

import argparse
import ctypes
import ctypes.wintypes
import sys
import time
import threading
from typing import Optional

try:
    ctypes.windll.shcore.SetProcessDpiAwareness(2)
except Exception:
    pass

import numpy as np
import mss

from config.settings import Settings, RunMode, load_settings
from capture.mss_capture import MSSCapture
from vision.pipeline import VisionPipeline
from tracking.tracker import Tracker, BalloonTrack
from decision.selector import TargetSelector, ClickAction
from input.sendinput import SendInputBackend
from input.coordinates import frame_to_screen
from telemetry.logger import TelemetryLogger, FrameTelemetry
from telemetry.timing import now_monotonic
from debug.visualizer import DebugVisualizer
from game_profiles.base import GameProfile, GameAction, GameType
from game_profiles import get_profile
from game_profiles.balloon_types import BalloonAction, TwinAction
from decision.mechanics import BalloonDecisionEngine
from behavior.model import BehavioralModel
from behavior.action_scheduler import ActionScheduler
from behavior.balloon_executor import BalloonExecutor


def _balloon_action_to_click_actions(action: BalloonAction | TwinAction) -> list[ClickAction]:
    targets = action.targets if isinstance(action, TwinAction) else (action,)
    result: list[ClickAction] = []
    for target in targets:
        metadata = dict(target.metadata)
        metadata.update({
            "color": target.color.value if target.color else "unknown",
            "target_type": target.target_type.value,
            "is_twin": target.is_twin,
            "is_special_sequence": target.is_special_sequence,
            "is_g": target.is_g,
            "target_width": target.target_width,
        })
        result.append(
            ClickAction(
                track_id=target.target_id if target.target_id is not None else -1,
                position=target.position,
                color=metadata["color"],
                confidence=float(metadata.get("confidence", 1.0)),
                reason=target.target_type.value,
                timestamp=now_monotonic(),
                metadata=metadata,
            )
        )
    return result


def _balloon_action_to_game_actions(action: BalloonAction | TwinAction) -> list[GameAction]:
    targets = action.targets if isinstance(action, TwinAction) else (action,)
    result: list[GameAction] = []
    for target in targets:
        metadata = dict(target.metadata)
        metadata.update({
            "color": target.color.value if target.color else "unknown",
            "target_type": target.target_type.value,
            "is_twin": target.is_twin,
            "is_special_sequence": target.is_special_sequence,
            "is_g": target.is_g,
        })
        result.append(
            GameAction(
                position=target.position,
                reason=target.target_type.value,
                timestamp=now_monotonic(),
                metadata=metadata,
            )
        )
    return result


def _game_action_to_click_action(a: GameAction) -> ClickAction:
    return ClickAction(
        track_id=-1,
        position=a.position,
        color=a.metadata.get("color", "unknown") if a.metadata else "unknown",
        confidence=a.metadata.get("confidence", 1.0) if a.metadata else 1.0,
        reason=a.reason,
        timestamp=a.timestamp,
        metadata=a.metadata,
    )


def _game_detection_to_track(det: dict, track_id: int) -> BalloonTrack:
    now = time.perf_counter_ns() / 1e9
    return BalloonTrack(
        id=track_id,
        position=det.get("center", (0, 0)),
        bbox=det.get("bbox", (0, 0, 0, 0)),
        color=det.get("color_name", "unknown"),
        has_outline=det.get("has_outline", False),
        confidence=det.get("color_confidence", det.get("confidence", 0.0)),
        first_seen=now,
        last_seen=now,
    )


class Application:
    def __init__(
        self,
        settings: Settings,
        game_profile: Optional[GameProfile] = None,
        duration: Optional[float] = None,
        startup_delay: float = 3.0,
    ) -> None:
        self._settings = settings
        self._running = True
        self._input_enabled = settings.input.enabled
        self._mode = settings.mode
        self._duration = duration
        self._startup_delay = startup_delay
        self._start_time = now_monotonic()
        self._game_profile = game_profile

        self._capture = MSSCapture(settings.capture)
        self._vision = VisionPipeline(settings)
        self._tracker = Tracker(settings)
        self._selector = TargetSelector(settings)
        self._input_backend = SendInputBackend() if settings.input.enabled else None
        self._telemetry = TelemetryLogger()
        self._visualizer = DebugVisualizer(settings)

        self._behavior = None
        self._action_scheduler = None
        self._balloon_decision = None
        self._balloon_executor = None
        self._balloon_frame_decision = None
        if settings.behavior.enabled:
            self._behavior = BehavioralModel(settings.behavior)
            self._action_scheduler = ActionScheduler()
            if self._game_profile and self._game_profile.game_type == GameType.BALLOON_GAME:
                self._balloon_executor = BalloonExecutor(self._behavior)
        if self._game_profile and self._game_profile.game_type == GameType.BALLOON_GAME:
            self._balloon_decision = BalloonDecisionEngine(settings)

        self._frame_count = 0
        self._total_clicks = 0
        self._fps_counter_start = now_monotonic()
        self._fps_frame_count = 0
        self._current_fps = 0.0

    def _setup_hotkeys(self) -> None:
        HK_F8 = 0x77
        HK_ESC = 0x1B
        WH_KEYBOARD_LL = 13

        def callback(nCode, wParam, lParam):
            if nCode >= 0 and wParam in (0x0100, 0x0104):
                vk = ctypes.cast(lParam, ctypes.POINTER(ctypes.c_uint)).contents.value
                if vk == HK_F8:
                    self._input_enabled = not self._input_enabled
                    print(f"[HOTKEY] Input {'ENABLED' if self._input_enabled else 'DISABLED'}")
                elif vk == HK_ESC:
                    print("[HOTKEY] ESC pressed, shutting down...")
                    self._running = False
            return user32.CallNextHookEx(None, nCode, wParam, ctypes.c_void_p(lParam))

        hook_proc = ctypes.WINFUNCTYPE(ctypes.c_int, ctypes.c_int, ctypes.wintypes.WPARAM, ctypes.wintypes.LPARAM)(callback)
        user32 = ctypes.windll.user32
        hook = user32.SetWindowsHookExW(WH_KEYBOARD_LL, hook_proc, None, 0)
        if not hook:
            print("[WARNING] Failed to install keyboard hook. F8/ESC hotkeys unavailable.")
            return

        msg = ctypes.wintypes.MSG()
        while self._running:
            if user32.PeekMessageW(ctypes.byref(msg), None, 0, 0, 0):
                user32.TranslateMessage(ctypes.byref(msg))
                user32.DispatchMessageW(ctypes.byref(msg))
            else:
                time.sleep(0.001)

        user32.UnhookWindowsHookEx(hook)

    def _process_frame(self, frame: np.ndarray):
        if self._game_profile is not None:
            detections = self._game_profile.detect(frame)
            if self._balloon_decision is not None:
                tracking_detections = self._game_profile.to_tracking_detections(detections)
                tracks = self._tracker.update(tracking_detections)
                current_detections = []
                for track in tracks.values():
                    if track.frames_missing != 0:
                        continue
                    observed = dict(track.metadata)
                    observed.update({
                        "target_id": track.id,
                        "center": track.position,
                        "bbox": track.bbox,
                        "color_name": track.color if track.color != "unknown" else observed.get("color_name"),
                        "color_confidence": track.confidence,
                        "confidence": track.confidence,
                    })
                    current_detections.append(observed)

                decision = self._balloon_decision.decide(current_detections)
                self._balloon_frame_decision = decision

                click_actions: list[ClickAction] = []
                game_actions: list[GameAction] = []
                for action in decision.actions:
                    click_actions.extend(_balloon_action_to_click_actions(action))
                    game_actions.extend(_balloon_action_to_game_actions(action))
                return detections, tracks, click_actions, game_actions

            game_actions = self._game_profile.select_targets(detections, frame.shape)
            click_actions = [_game_action_to_click_action(a) for a in game_actions]
            tracks_dict = {}
            for i, det in enumerate(detections):
                track = _game_detection_to_track(det, i)
                tracks_dict[track.id] = track
            return detections, tracks_dict, click_actions, game_actions
        else:
            detections, vision_ms = self._vision.process(frame)
            tracks = self._tracker.update(detections)
            click_actions = self._selector.select(tracks)
            return detections, tracks, click_actions, None

    def _run_benchmark(self) -> None:
        print("Running BENCHMARK mode - no mouse input, measuring raw pipeline speed")
        if self._duration:
            print(f"Running for {self._duration} seconds")
        if self._game_profile:
            print(f"Game profile: {self._game_profile.name}")
        print("Press ESC to stop\n")
        sys.stdout.flush()

        durations: list[float] = []
        frame_times: list[float] = []

        try:
            while self._running:
                if self._duration and (now_monotonic() - self._start_time) >= self._duration:
                    print(f"\nDuration limit reached ({self._duration}s)")
                    break
                t_start = now_monotonic()

                frame, capture_ts = self._capture.grab()
                t_after_capture = now_monotonic()

                detections, tracks, actions, game_actions = self._process_frame(frame)
                t_after_decision = now_monotonic()

                self._frame_count += 1
                self._fps_frame_count += 1
                elapsed = now_monotonic() - self._fps_counter_start
                if elapsed >= 1.0:
                    self._current_fps = self._fps_frame_count / elapsed
                    self._fps_frame_count = 0
                    self._fps_counter_start = now_monotonic()

                proc_ms = (t_after_decision - t_start) * 1000
                e2e_ms = (t_after_decision - capture_ts) * 1000
                durations.append(proc_ms)
                frame_times.append(e2e_ms)

                if self._frame_count % 100 == 0:
                    print(
                        f"Frame {self._frame_count:6d} | "
                        f"FPS: {self._current_fps:6.1f} | "
                        f"Proc: {proc_ms:6.2f}ms | "
                        f"Detections: {len(detections)} | "
                        f"Tracks: {len(tracks)}"
                    )
                    sys.stdout.flush()

                if self._settings.debug.show_detections:
                    vis = self._visualizer.render(
                        frame, detections, tracks, actions,
                        input_enabled=False, fps=self._current_fps,
                    )
                key = self._visualizer.show(vis)
                if key == 27:
                    if self._action_scheduler and not self._action_scheduler.is_idle:
                        self._action_scheduler.cancel()
                    self._running = False
                    break

        except KeyboardInterrupt:
            if self._action_scheduler and not self._action_scheduler.is_idle:
                self._action_scheduler.cancel()

        self._print_benchmark_results(durations, frame_times)

    def _run_dry_run(self) -> None:
        print("Running DRY-RUN mode - detecting targets, no mouse clicks")
        if self._duration:
            print(f"Running for {self._duration} seconds")
        if self._game_profile:
            print(f"Game profile: {self._game_profile.name}")
        print("Press ESC to stop\n")
        sys.stdout.flush()

        try:
            while self._running:
                if self._duration and (now_monotonic() - self._start_time) >= self._duration:
                    print(f"\nDuration limit reached ({self._duration}s)")
                    break
                t_start = now_monotonic()

                frame, capture_ts = self._capture.grab()
                detections, tracks, actions, game_actions = self._process_frame(frame)

                self._frame_count += 1
                self._fps_frame_count += 1
                elapsed = now_monotonic() - self._fps_counter_start
                if elapsed >= 1.0:
                    self._current_fps = self._fps_frame_count / elapsed
                    self._fps_frame_count = 0
                    self._fps_counter_start = now_monotonic()

                proc_ms = (now_monotonic() - t_start) * 1000

                telem = FrameTelemetry(
                    frame_id=self._frame_count,
                    capture_timestamp=capture_ts,
                    detection_timestamp=capture_ts,
                    decision_timestamp=now_monotonic(),
                    input_timestamp=None,
                    processing_duration=proc_ms,
                    end_to_end_latency=(now_monotonic() - capture_ts) * 1000,
                    detections_count=len(detections),
                    tracks_count=len(tracks),
                    clicks_sent=0,
                    behavioral=self._telemetry.behavioral.extract(len(tracks)),
                )
                self._telemetry.log_frame(telem)

                if self._frame_count % 60 == 0:
                    print(
                        f"Frame {self._frame_count:6d} | "
                        f"FPS: {self._current_fps:6.1f} | "
                        f"Proc: {proc_ms:6.2f}ms | "
                        f"Detections: {len(detections)} | "
                        f"Tracks: {len(tracks)} | "
                        f"Actions: {len(actions)}"
                    )
                    sys.stdout.flush()

                vis = self._visualizer.render(
                    frame, detections, tracks, actions,
                    input_enabled=False, fps=self._current_fps,
                )
                key = self._visualizer.show(vis)
                if key == 27:
                    self._running = False
                    break

        except KeyboardInterrupt:
            pass

        self._telemetry.save()
        print(f"\nTelemetry saved to {self._telemetry._output_dir}")

    def _run_live(self) -> None:
        if not self._settings.input.enabled:
            print("WARNING: Live mode requested but input.enabled is False in config.")
            print("Setting input.enabled = True for live mode.")
            self._settings.input.enabled = True
            self._input_enabled = True

        self._input_backend = SendInputBackend()

        print("Running LIVE mode - detecting AND clicking targets")
        if self._game_profile:
            print(f"Game profile: {self._game_profile.name}")
        if self._behavior:
            print(f"Behavioral model: enabled (seed={self._behavior.config.seed})")
        print("F8 = Toggle click on/off | ESC = Emergency stop")

        if self._startup_delay > 0:
            print(f"\nStarting in {self._startup_delay:.1f}s — switch to your game window now...")
            delay_end = now_monotonic() + self._startup_delay
            last_printed = -1
            while self._running and now_monotonic() < delay_end:
                remaining = delay_end - now_monotonic()
                secs = int(remaining) + 1
                if secs != last_printed and secs > 0:
                    print(f"  {secs}s remaining...")
                    last_printed = secs
                time.sleep(0.05)
            if not self._running:
                return
            print("GO — input is now ACTIVE\n")
            self._start_time = now_monotonic()
            
        else:
            print("Input is ACTIVE. Move mouse to safe area if needed.\n")

        if self._behavior and not self._behavior._session_started:
            self._behavior.start_session()

        try:
            while self._running:
                if self._duration and (now_monotonic() - self._start_time) >= self._duration:
                    print(f"\nDuration limit reached ({self._duration}s)")
                    break

                t_start = now_monotonic()

                frame, capture_ts = self._capture.grab()
                detections, tracks, actions, game_actions = self._process_frame(frame)

                self._frame_count += 1
                self._fps_frame_count += 1
                elapsed = now_monotonic() - self._fps_counter_start
                if elapsed >= 1.0:
                    self._current_fps = self._fps_frame_count / elapsed
                    self._fps_frame_count = 0
                    self._fps_counter_start = now_monotonic()

                clicks_this_frame = 0
                input_ts = None
                action_ts = None
                region = self._settings.capture.region

                if self._balloon_executor and self._balloon_frame_decision is not None:
                    self._balloon_executor.set_mouse_position(
                        (self._input_backend._last_x, self._input_backend._last_y)
                    )
                    execution = self._balloon_executor.tick(
                        self._balloon_frame_decision.actions,
                        self._balloon_frame_decision.g_action,
                        self._input_backend,
                        region.left,
                        region.top,
                        self._input_enabled,
                    )
                    clicks_this_frame = execution.clicks_sent
                    if execution.input_timestamp is not None:
                        input_ts = execution.input_timestamp
                    action_ts = execution.action_timestamps

                    if execution.completed_primary_action is not None:
                        self._balloon_decision.complete_action(
                            execution.completed_primary_action
                        )

                    self._total_clicks += execution.clicks_sent

                elif self._behavior and self._action_scheduler:
                    completed = self._action_scheduler.tick(self._input_backend)
                    if completed is not None:
                        self._total_clicks += 1
                        clicks_this_frame += 1
                        input_ts = completed.click_up
                        action_ts = completed
                        self._input_backend._last_x = self._action_scheduler.active_action.target_x if self._action_scheduler.active_action else self._input_backend._last_x
                        self._input_backend._last_y = self._action_scheduler.active_action.target_y if self._action_scheduler.active_action else self._input_backend._last_y
                        self._action_scheduler.clear()

                    if self._action_scheduler.is_idle and self._input_enabled and actions:
                        action = actions[0]
                        px, py = action.position
                        screen_x, screen_y = frame_to_screen(px, py, region.left, region.top)

                        behavioral_action = self._behavior.plan_action(
                            current_x=self._input_backend._last_x,
                            current_y=self._input_backend._last_y,
                            target_x=float(screen_x),
                            target_y=float(screen_y),
                            target_width=float(action.metadata.get("area", 5000) ** 0.5) if action.metadata else 5000.0,
                            target_metadata=action.metadata,
                        )

                        if behavioral_action.is_miss:
                            miss_offset_x = self._behavior.rng.normal(0, 50)
                            miss_offset_y = self._behavior.rng.normal(0, 50)
                            screen_x = int(screen_x + miss_offset_x)
                            screen_y = int(screen_y + miss_offset_y)

                        self._action_scheduler.start_action(
                            behavioral_action, screen_x, screen_y
                        )

                        if self._game_profile and game_actions:
                            self._game_profile.on_click_sent(game_actions[0])

                elif self._input_enabled and actions:
                    for i, action in enumerate(actions):
                        px, py = action.position
                        screen_x, screen_y = frame_to_screen(px, py, region.left, region.top)
                        ts = self._input_backend.click_screen(screen_x, screen_y)
                        clicks_this_frame += 1
                        self._total_clicks += 1

                        if self._game_profile and game_actions and i < len(game_actions):
                            self._game_profile.on_click_sent(game_actions[i])

                        if i > 0 and actions:
                            pair_delta = (ts - actions[i - 1].timestamp) if actions[i - 1].timestamp else None
                            if pair_delta is not None:
                                self._telemetry.behavioral.record_pair_click(pair_delta)

                        if self._settings.input.click_delay_ms > 0:
                            time.sleep(self._settings.input.click_delay_ms / 1000.0)

                    input_ts = ts if clicks_this_frame > 0 else None

                proc_ms = (now_monotonic() - t_start) * 1000

                telem = FrameTelemetry(
                    frame_id=self._frame_count,
                    capture_timestamp=capture_ts,
                    detection_timestamp=capture_ts,
                    decision_timestamp=now_monotonic(),
                    input_timestamp=input_ts,
                    processing_duration=proc_ms,
                    end_to_end_latency=(now_monotonic() - capture_ts) * 1000,
                    detections_count=len(detections),
                    tracks_count=len(tracks),
                    clicks_sent=clicks_this_frame,
                    behavioral=self._telemetry.behavioral.extract(len(tracks)),
                    action_timestamps=action_ts,
                )
                self._telemetry.log_frame(telem)

                if self._frame_count % 60 == 0:
                    print(
                        f"Frame {self._frame_count:6d} | "
                        f"FPS: {self._current_fps:6.1f} | "
                        f"Proc: {proc_ms:6.2f}ms | "
                        f"Clicks: {self._total_clicks:5d} | "
                        f"Input: {'ON' if self._input_enabled else 'OFF'}"
                    )
                    sys.stdout.flush()

                vis = self._visualizer.render(
                    frame, detections, tracks, actions,
                    input_enabled=self._input_enabled, fps=self._current_fps,
                )
                key = self._visualizer.show(vis)
                if key == 27:
                    self._running = False
                    break

        except KeyboardInterrupt:
            pass

        self._telemetry.save()
        print(f"\nTelemetry saved to {self._telemetry._output_dir}")

    def _print_benchmark_results(self, durations: list[float], frame_times: list[float]) -> None:
        print("\n" + "=" * 60)
        print("BENCHMARK RESULTS")
        print("=" * 60)
        print(f"Total frames:    {len(durations)}")
        print(f"Total clicks:    {self._total_clicks}")

        if durations:
            arr = np.array(durations)
            print(f"\nProcessing time (ms):")
            print(f"  Mean:   {np.mean(arr):8.3f}")
            print(f"  Median: {np.median(arr):8.3f}")
            print(f"  Min:    {np.min(arr):8.3f}")
            print(f"  Max:    {np.max(arr):8.3f}")
            print(f"  P95:    {np.percentile(arr, 95):8.3f}")
            print(f"  P99:    {np.percentile(arr, 99):8.3f}")
            print(f"  Std:    {np.std(arr):8.3f}")

        if frame_times:
            arr_ft = np.array(frame_times)
            print(f"\nEnd-to-end latency (ms):")
            print(f"  Mean:   {np.mean(arr_ft):8.3f}")
            print(f"  Median: {np.median(arr_ft):8.3f}")
            print(f"  P95:    {np.percentile(arr_ft, 95):8.3f}")
            print(f"  P99:    {np.percentile(arr_ft, 99):8.3f}")

        if durations:
            print(f"\nEstimated max FPS: {1000.0 / np.mean(arr):.1f}")

        print("=" * 60)

    def run(self) -> None:
        hook_thread = threading.Thread(target=self._setup_hotkeys, daemon=True)
        hook_thread.start()

        if self._mode == RunMode.BENCHMARK:
            self._run_benchmark()
        elif self._mode == RunMode.DRY_RUN:
            self._run_dry_run()
        elif self._mode == RunMode.LIVE:
            self._run_live()

        self._capture.close()
        self._visualizer.close()
        if self._input_backend:
            self._input_backend.close()

        print(f"\nSession complete. {self._frame_count} frames processed, {self._total_clicks} clicks sent.")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Balloon Game Automation")
    parser.add_argument(
        "--mode",
        choices=["benchmark", "dry-run", "live"],
        default=None,
        help="Run mode (overrides config file)",
    )
    parser.add_argument(
        "--game",
        type=str,
        default=None,
        choices=["cookie", "balloons"],
        help="Game profile to use (cookie=Cookie Clicker, balloons=Balloon Game)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config JSON file",
    )
    parser.add_argument(
        "--region",
        type=str,
        default=None,
        help="Capture region as 'top,left,width,height'",
    )
    parser.add_argument(
        "--fullscreen",
        action="store_true",
        help="Capture the entire primary monitor (auto-detected from MSS)",
    )
    parser.add_argument(
        "--no-gui",
        action="store_true",
        help="Run without debug visualization window",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=None,
        help="Run for N seconds then stop (useful for benchmarking)",
    )
    parser.add_argument(
        "--startup-delay",
        type=float,
        default=3.0,
        help="Seconds to wait before sending input in LIVE mode (default: 3.0, 0 to disable)",
    )
    return parser.parse_args(argv)


def main():
    args = parse_args()

    settings = load_settings(args.config)

    if args.mode:
        settings.mode = RunMode(args.mode)

    game_profile = None
    if args.game:
        game_profile = get_profile(args.game)

    if args.fullscreen:
        try:
            sct = mss.MSS()
            mon = sct.monitors[1]
            settings.capture.region.left = mon["left"]
            settings.capture.region.top = mon["top"]
            settings.capture.region.width = mon["width"]
            settings.capture.region.height = mon["height"]
            sct.close()
        except Exception:
            settings.capture.region.left = 0
            settings.capture.region.top = 0
            settings.capture.region.width = 1920
            settings.capture.region.height = 1080
            print("[WARNING] MSS monitor detection failed, falling back to 1920x1080")
    elif args.region:
        parts = [int(x.strip()) for x in args.region.split(",")]
        settings.capture.region.top = parts[0]
        settings.capture.region.left = parts[1]
        settings.capture.region.width = parts[2]
        settings.capture.region.height = parts[3]

    if args.no_gui:
        settings.debug.show_detections = False

    print(f"Mode: {settings.mode.value}")
    if game_profile:
        print(f"Game: {game_profile.name}")
    print(f"Region: {settings.capture.region.left},{settings.capture.region.top} "
          f"{settings.capture.region.width}x{settings.capture.region.height}")
    print(f"Input enabled: {settings.input.enabled}")
    if settings.mode == RunMode.LIVE and args.startup_delay > 0:
        print(f"Startup delay: {args.startup_delay}s")
    print()

    app = Application(settings, game_profile=game_profile, duration=args.duration, startup_delay=args.startup_delay)
    app.run()


if __name__ == "__main__":
    main()
