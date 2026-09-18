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
from telemetry.outcome import SessionOutcome
from vision.feedback import FeedbackReader
from debug.visualizer import DebugVisualizer
from game_profiles.base import GameProfile, GameAction, GameType
from game_profiles import get_profile
from game_profiles.balloon_types import BalloonAction, TwinAction
from decision.mechanics import BalloonDecisionEngine
from behavior.model import BehavioralModel
from behavior.action_scheduler import ActionScheduler
from behavior.balloon_executor import BalloonExecutor
from diagnostics.downstream import DownstreamDiagnostic


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
        fast_execution: bool = False,
    ) -> None:
        self._settings = settings
        self._running = True
        self._input_enabled = settings.input.enabled
        self._mode = settings.mode
        self._duration = duration
        self._startup_delay = startup_delay
        self._fast_execution = fast_execution
        self._start_time = now_monotonic()
        self._game_profile = game_profile

        self._capture = MSSCapture(settings.capture)
        self._vision = VisionPipeline(settings)
        self._tracker = Tracker(settings)
        self._selector = TargetSelector(settings)
        self._diagnostic = DownstreamDiagnostic() if game_profile and game_profile.game_type == GameType.BALLOON_GAME else None
        self._input_backend = SendInputBackend(diagnostic=self._diagnostic) if settings.input.enabled else None
        self._telemetry = TelemetryLogger()
        self._feedback_reader = FeedbackReader(
            combo_roi=settings.feedback.combo_roi,
            score_roi=settings.feedback.score_roi,
            pop_roi=settings.feedback.pop_roi,
            change_threshold=settings.feedback.change_threshold,
        )
        self._last_feedback = None
        self._visualizer = DebugVisualizer(settings)

        self._behavior = None
        self._action_scheduler = None
        self._balloon_decision = None
        self._balloon_executor = None
        self._balloon_frame_decision = None
        if settings.behavior.enabled:
            self._behavior = BehavioralModel(settings.behavior)
            self._telemetry.set_session_metadata(
                behavioral_seed=self._behavior.behavioral_seed,
                human_profile=self._behavior.profile.as_dict(),
                feedback_signals="screen-only; unvalidated UI signals remain UNOBSERVABLE THROUGH CURRENT INTERFACE",
            )
            self._action_scheduler = ActionScheduler()
            if self._game_profile and self._game_profile.game_type == GameType.BALLOON_GAME:
                self._balloon_executor = BalloonExecutor(
                    self._behavior,
                    diagnostic=self._diagnostic,
                    fast_execution=self._fast_execution,
                )
        if self._game_profile and self._game_profile.game_type == GameType.BALLOON_GAME:
            self._balloon_decision = BalloonDecisionEngine(settings)

        self._frame_count = 0
        self._total_clicks = 0
        self._fps_counter_start = now_monotonic()
        self._fps_frame_count = 0
        self._current_fps = 0.0
        self._last_frame_stage_timings = {}

    def _setup_hotkeys(self) -> None:
        HK_RSHIFT = 0xA1
        HK_ESC = 0x1B
        WH_KEYBOARD_LL = 13

        def callback(nCode, wParam, lParam):
            if nCode >= 0 and wParam in (0x0100, 0x0104):
                vk = ctypes.cast(lParam, ctypes.POINTER(ctypes.c_uint)).contents.value
                if vk == HK_RSHIFT:
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
            print("[WARNING] Failed to install keyboard hook. Right Shift/ESC hotkeys unavailable.")
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
        t0 = now_monotonic()
        self._last_feedback = self._feedback_reader.observe(frame, t0)
        feedback_ms = (now_monotonic() - t0) * 1000.0
        if self._game_profile is not None:
            t0 = now_monotonic()
            detections = self._game_profile.detect(frame)
            vision_ms = (now_monotonic() - t0) * 1000.0
            if self._balloon_decision is not None:
                t0 = now_monotonic()
                tracking_detections = self._game_profile.to_tracking_detections(detections)
                tracks = self._tracker.update(tracking_detections)
                tracking_ms = (now_monotonic() - t0) * 1000.0
                current_detections = []
                for track in tracks.values():
                    if track.frames_missing != 0:
                        continue
                    observed = dict(track.metadata)
                    observed.update({
                        "target_id": track.id,
                        "first_seen_timestamp": track.first_seen,
                        "last_seen_timestamp": track.last_seen,
                        "center": track.position,
                        "bbox": track.bbox,
                        "color_name": track.color if track.color != "unknown" else observed.get("color_name"),
                        "color_confidence": track.confidence,
                        "confidence": track.confidence,
                    })
                    current_detections.append(observed)

                t0 = now_monotonic()
                decision = self._balloon_decision.decide(current_detections)
                decision_ms = (now_monotonic() - t0) * 1000.0
                self._balloon_frame_decision = decision
                self._last_frame_stage_timings = {
                    "feedback_ms": feedback_ms,
                    "vision_ms": vision_ms,
                    "tracking_ms": tracking_ms,
                    "decision_ms": decision_ms,
                }
                if self._diagnostic:
                    self._diagnostic.log("tracking_decision", frame_id=self._frame_count + 1, detections=[{"type": d.get("target_type"), "id": d.get("target_id"), "color": d.get("color_name"), "center": d.get("center")} for d in current_detections], detection_counts=self._count_types(current_detections), actionable=[self._action_info_for_diag(a) for a in decision.actions], actionable_counts=self._count_types_from_actions(decision.actions), g_action=self._action_info_for_diag(decision.g_action), multicolor_specials=[self._action_info_for_diag(a) for a in decision.multicolor_specials], notifications=[self._action_info_for_diag(a) for a in decision.notifications], mode=decision.mode.value, special_step=decision.special_step.value)

                click_actions: list[ClickAction] = []
                game_actions: list[GameAction] = []
                for action in decision.actions:
                    click_actions.extend(_balloon_action_to_click_actions(action))
                    game_actions.extend(_balloon_action_to_game_actions(action))
                return detections, tracks, click_actions, game_actions

            t0 = now_monotonic()
            game_actions = self._game_profile.select_targets(detections, frame.shape)
            self._last_frame_stage_timings = {
                "feedback_ms": feedback_ms,
                "vision_ms": vision_ms,
                "tracking_ms": 0.0,
                "decision_ms": (now_monotonic() - t0) * 1000.0,
            }
            click_actions = [_game_action_to_click_action(a) for a in game_actions]
            tracks_dict = {}
            for i, det in enumerate(detections):
                track = _game_detection_to_track(det, i)
                tracks_dict[track.id] = track
            return detections, tracks_dict, click_actions, game_actions
        else:
            t0 = now_monotonic()
            detections, _vision_internal_ms = self._vision.process(frame)
            vision_ms = (now_monotonic() - t0) * 1000.0
            t0 = now_monotonic()
            tracks = self._tracker.update(detections)
            tracking_ms = (now_monotonic() - t0) * 1000.0
            t0 = now_monotonic()
            click_actions = self._selector.select(tracks)
            decision_ms = (now_monotonic() - t0) * 1000.0
            self._last_frame_stage_timings = {
                "feedback_ms": feedback_ms,
                "vision_ms": vision_ms,
                "tracking_ms": tracking_ms,
                "decision_ms": decision_ms,
            }
            return detections, tracks, click_actions, None

    @staticmethod
    def _count_types(detections):
        counts = {}
        for d in detections:
            key = d.get("target_type")
            counts[key] = counts.get(key, 0) + 1
        return counts

    @staticmethod
    def _count_types_from_actions(actions):
        counts = {}
        for action in actions:
            key = type(action).__name__ if isinstance(action, TwinAction) else action.target_type.value
            counts[key] = counts.get(key, 0) + 1
        return counts

    @staticmethod
    def _action_info_for_diag(action):
        if action is None:
            return None
        if isinstance(action, TwinAction):
            return {"type": "TwinAction", "targets": [Application._action_info_for_diag(t) for t in action.targets]}
        return {"type": action.target_type.value, "id": action.target_id, "color": action.color.value if action.color else None, "position": list(action.position), "width": action.target_width}

    @staticmethod
    def _active_info_for_diag(active):
        if active is None:
            return None
        return {"phase": active.phase.name, "target_x": active.target_x, "target_y": active.target_y, "intended_target_x": active.intended_target_x, "intended_target_y": active.intended_target_y}

    def _twin_active_for_diag(self):
        active = self._balloon_executor.twin_scheduler.active_actions
        if active is None:
            return None
        return [self._active_info_for_diag(a) for a in active]

    def _behavior_gate_diag(self, duration_s):
        if not self._behavior:
            return {}
        elapsed = self._behavior.session_elapsed_s()
        cutoff = self._behavior.activity_cutoff_s()
        deadline = None if not duration_s or duration_s <= 0 else duration_s - cutoff
        return {"session_started": self._behavior._session_started, "session_elapsed_s": elapsed, "start_delay_s": self._behavior.start_delay_s, "actions_ready": self._behavior.actions_ready(), "activity_cutoff_s": cutoff, "final_activity_deadline_s": deadline, "final_activity_cutoff_active": bool(deadline is not None and elapsed >= deadline)}

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

        self._input_backend = SendInputBackend(diagnostic=self._diagnostic)
        if self._balloon_executor:
            self._balloon_executor.set_mouse_position(
                (self._input_backend._last_x, self._input_backend._last_y)
            )

        print("Running LIVE mode - detecting AND clicking targets")
        if self._game_profile:
            print(f"Game profile: {self._game_profile.name}")
        if self._behavior:
            print(f"Behavioral model: enabled (seed={self._behavior.behavioral_seed})")
            print(f"Human profile: {self._behavior.profile.as_dict()}")
        if self._fast_execution:
            print("Balloon execution: FAST (immediate SendInput dispatch)")
        print("RSHIFT = Toggle click on/off | ESC = Emergency stop")

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

        if self._behavior and not self._behavior._session_started and not self._fast_execution:
            self._behavior.start_session()
            print(f"Sampled behavioral start-delay parameter: {self._behavior.start_delay_s:.3f}s")

        try:
            while self._running:
                if self._duration and (now_monotonic() - self._start_time) >= self._duration:
                    print(f"\nDuration limit reached ({self._duration}s)")
                    break
                if self._behavior and not self._fast_execution and self._behavior.should_early_exit(self._duration):
                    print("\nBehavioral model sampled an early session exit.")
                    break

                t_start = now_monotonic()
                if self._diagnostic:
                    self._diagnostic.begin_frame()

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
                    new_actions_allowed = (
                        self._fast_execution
                        or not self._behavior
                        or self._behavior.should_start_new_action(self._duration)
                    )
                    if self._diagnostic:
                        self._diagnostic.log("frame_to_executor", frame_id=self._frame_count, passed=new_actions_allowed, decision_actions=[self._action_info_for_diag(a) for a in self._balloon_frame_decision.actions], passed_actions=[self._action_info_for_diag(a) for a in (self._balloon_frame_decision.actions if new_actions_allowed else ())], g_passed=new_actions_allowed and self._balloon_frame_decision.g_action is not None, notification_count=len(self._balloon_frame_decision.notifications) if new_actions_allowed else 0, multicolor_count=len(self._balloon_frame_decision.multicolor_specials) if new_actions_allowed else 0, behavioral_gate=self._behavior_gate_diag(self._duration))
                    executor_t0 = now_monotonic()
                    execution = self._balloon_executor.tick(
                        self._balloon_frame_decision.actions if new_actions_allowed else (),
                        self._balloon_frame_decision.g_action if new_actions_allowed else None,
                        self._input_backend,
                        region.left,
                        region.top,
                        self._input_enabled,
                        self._balloon_frame_decision.notifications if new_actions_allowed else (),
                        self._balloon_frame_decision.multicolor_specials if new_actions_allowed else (),
                        duration_s=self._duration,
                    )
                    executor_ms = (now_monotonic() - executor_t0) * 1000.0
                    if self._diagnostic:
                        self._diagnostic.log("executor_result", frame_id=self._frame_count, clicks_sent=execution.clicks_sent, completed_primary=self._action_info_for_diag(execution.completed_primary_action), twin_completed=execution.twin_completion is not None, g_completed=execution.g_click_completed, primary_active=self._active_info_for_diag(self._balloon_executor.primary_scheduler.active_action), twin_active=self._twin_active_for_diag(), g_active=self._active_info_for_diag(self._balloon_executor.g_scheduler.active_action), idle_until=self._balloon_executor._idle_until)
                    clicks_this_frame = execution.clicks_sent
                    if execution.input_timestamp is not None:
                        input_ts = execution.input_timestamp
                    action_ts = execution.action_timestamps

                    if execution.completed_primary_action is not None:
                        self._balloon_decision.complete_action(
                            execution.completed_primary_action
                        )

                    self._total_clicks += execution.clicks_sent
                    for outcome in execution.outcomes:
                        if self._last_feedback is not None:
                            outcome.feedback_confidence = self._last_feedback.confidence
                            outcome.observation_notes.extend(self._last_feedback.notes)
                        self._telemetry.record_outcome(outcome)

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
                telemetry_t0 = now_monotonic()
                self._telemetry.log_frame(telem)
                telemetry_ms = (now_monotonic() - telemetry_t0) * 1000.0

                gui_t0 = now_monotonic()
                vis = self._visualizer.render(
                    frame, detections, tracks, actions,
                    input_enabled=self._input_enabled, fps=self._current_fps,
                )
                gui_render_ms = (now_monotonic() - gui_t0) * 1000.0
                gui_show_t0 = now_monotonic()
                key = self._visualizer.show(vis)
                gui_show_ms = (now_monotonic() - gui_show_t0) * 1000.0

                if self._diagnostic:
                    self._diagnostic.log(
                        "frame_timing",
                        capture_ms=self._capture.last_timing_ms.get("capture", 0.0),
                        numpy_ms=self._capture.last_timing_ms.get("numpy", 0.0),
                        feedback_ms=self._last_frame_stage_timings.get("feedback_ms", 0.0),
                        vision_ms=self._last_frame_stage_timings.get("vision_ms", 0.0),
                        tracking_ms=self._last_frame_stage_timings.get("tracking_ms", 0.0),
                        decision_ms=self._last_frame_stage_timings.get("decision_ms", 0.0),
                        behavior_ms=getattr(self._balloon_executor, "last_behavior_plan_ms", 0.0),
                        executor_ms=locals().get("executor_ms", 0.0),
                        telemetry_ms=telemetry_ms,
                        gui_render_ms=gui_render_ms,
                        gui_show_ms=gui_show_ms,
                        diagnostic_ms=self._diagnostic.frame_log_duration_ms,
                        frame_total_ms=(now_monotonic() - t_start) * 1000.0,
                        vision_profile=(
                            self._game_profile.last_profile
                            if self._game_profile is not None
                            and self._game_profile.game_type == GameType.BALLOON_GAME
                            else None
                        ),
                    )

                if key == 27:
                    self._running = False
                    break

                if self._frame_count % 60 == 0:
                    print(
                        f"Frame {self._frame_count:6d} | "
                        f"FPS: {self._current_fps:6.1f} | "
                        f"Proc: {proc_ms:6.2f}ms | "
                        f"Clicks: {self._total_clicks:5d} | "
                        f"Input: {'ON' if self._input_enabled else 'OFF'}"
                    )
                    sys.stdout.flush()

        except KeyboardInterrupt:
            pass
        finally:
            if self._balloon_executor is not None and self._input_backend is not None:
                for outcome in self._balloon_executor.shutdown(
                    self._input_backend,
                    reason="session_end",
                ):
                    if self._last_feedback is not None:
                        outcome.feedback_confidence = self._last_feedback.confidence
                        outcome.observation_notes.extend(self._last_feedback.notes)
                    self._telemetry.record_outcome(outcome)

        self._telemetry.save()
        print(f"\nTelemetry saved to {self._telemetry._output_dir}")
        if self._diagnostic:
            self._diagnostic.close()
            print(f"Diagnostic trace saved to {self._diagnostic.path}")

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
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Balloon mode: bypass behavioral reaction/movement/click delays and dispatch input immediately",
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

    app = Application(
        settings,
        game_profile=game_profile,
        duration=args.duration,
        startup_delay=args.startup_delay,
        fast_execution=args.fast,
    )
    app.run()


if __name__ == "__main__":
    main()
