from __future__ import annotations

import numpy as np
import cv2

from config.settings import Settings, CaptureConfig
from tracking.tracker import BalloonTrack


COLOR_BGR = {
    "red": (0, 0, 255),
    "green": (0, 255, 0),
    "blue": (255, 0, 0),
    "yellow": (0, 255, 255),
    "white": (255, 255, 255),
    "unknown": (128, 128, 128),
}

WINDOW_NAME = "Automation Debug"
WINDOW_W = 480
WINDOW_H = 360


def _get(det, key, default=None):
    if isinstance(det, dict):
        return det.get(key, default)
    return getattr(det, key, default)


class DebugVisualizer:
    def __init__(self, settings: Settings) -> None:
        self._config = settings.debug
        self._capture_config = settings.capture
        self._window_created = False

    def _ensure_window(self) -> None:
        if self._window_created:
            return
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW_NAME, WINDOW_W, WINDOW_H)
        self._position_window()
        self._window_created = True

    def _position_window(self) -> None:
        r = self._capture_config.region
        cap_right = r.left + r.width
        cap_bottom = r.top + r.height

        screen_w = 1920
        screen_h = 1080
        try:
            import mss
            sct = mss.MSS()
            mon = sct.monitors[1]
            screen_w = mon["width"]
            screen_h = mon["height"]
            sct.close()
        except Exception:
            pass

        if cap_right + WINDOW_W + 10 <= screen_w:
            win_x = cap_right + 10
            win_y = r.top
        elif r.left - WINDOW_W - 10 >= 0:
            win_x = r.left - WINDOW_W - 10
            win_y = r.top
        elif cap_bottom + WINDOW_H + 40 <= screen_h:
            win_x = r.left
            win_y = cap_bottom + 40
        elif r.top - WINDOW_H - 40 >= 0:
            win_x = r.left
            win_y = r.top - WINDOW_H - 40
        else:
            win_x = 0
            win_y = 0

        cv2.moveWindow(WINDOW_NAME, win_x, win_y)

    def render(
        self,
        frame: np.ndarray,
        detections,
        tracks: dict[int, BalloonTrack],
        actions,
        input_enabled: bool,
        fps: float = 0.0,
    ) -> np.ndarray:
        vis = frame.copy()

        if self._config.show_detections:
            for det in detections:
                x, y, w, h = _get(det, "bbox", (0, 0, 0, 0))
                color_name = _get(det, "color_name", _get(det, "color", "unknown"))
                color = COLOR_BGR.get(color_name, (128, 128, 128))

                if self._config.show_bounding_boxes:
                    outline = _get(det, "outline", None)
                    has_outline = outline.has_outline if outline else _get(det, "has_outline", False)
                    thickness = 2 if not has_outline else 3
                    cv2.rectangle(vis, (x, y), (x + w, y + h), color, thickness)

                if self._config.show_centers:
                    cx, cy = _get(det, "center", (x + w // 2, y + h // 2))
                    cv2.circle(vis, (cx, cy), 4, color, -1)

                if self._config.show_colors:
                    conf = _get(det, "color_confidence", _get(det, "confidence", 0.0))
                    label = f"{color_name} {conf:.2f}"
                    cv2.putText(vis, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

                if self._config.show_outline:
                    outline = _get(det, "outline", None)
                    has_outline = outline.has_outline if outline else _get(det, "has_outline", False)
                    if has_outline:
                        cv2.putText(vis, "OUTLINE", (x, y + h + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 0), 1)

        if self._config.show_track_ids:
            for tid, track in tracks.items():
                cx, cy = track.position
                label = f"ID:{tid}"
                cv2.putText(vis, label, (cx + 5, cy - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

        for action in actions:
            px, py = _get(action, "position", (0, 0))
            cv2.drawMarker(vis, (px, py), (0, 255, 255), cv2.MARKER_CROSS, 10, 2)

        mode_color = (0, 255, 0) if input_enabled else (0, 0, 255)
        mode_text = "LIVE" if input_enabled else "DRY-RUN"
        cv2.putText(vis, f"Mode: {mode_text}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, mode_color, 2)
        cv2.putText(vis, f"FPS: {fps:.1f}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(vis, f"Tracks: {len(tracks)}  Actions: {len(actions)}", (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.putText(vis, "Right Shift=Toggle Click  ESC=Quit", (10, vis.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)

        return vis

    def show(self, frame: np.ndarray) -> int:
        self._ensure_window()
        cv2.imshow(WINDOW_NAME, frame)
        key = cv2.waitKey(1) & 0xFF
        try:
            if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                return 27
        except cv2.error:
            return 27
        return key

    def close(self) -> None:
        if self._window_created:
            try:
                cv2.destroyWindow(WINDOW_NAME)
            except cv2.error:
                pass
            self._window_created = False
