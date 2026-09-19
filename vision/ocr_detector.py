from __future__ import annotations

import re
import threading
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np
import pytesseract


@dataclass(frozen=True)
class OCRWord:
    text: str
    confidence: float
    bbox: tuple[int, int, int, int]

    @property
    def normalized(self) -> str:
        return self.text.strip().lower()

    @property
    def is_numeric(self) -> bool:
        return bool(re.fullmatch(r"[0-9][0-9.,:%+\\-]*", self.normalized))

    @property
    def is_twin_text(self) -> bool:
        return self.normalized in {"pop", "together"} or "pop together" in self.normalized


@dataclass(frozen=True)
class OCRSnapshot:
    words: tuple[OCRWord, ...]
    completed_frame_id: int
    age_frames: int


class OCRDetector:
    """Asynchronous OCR safety/semantic layer for the game viewport.

    The game is rendered in a fixed central viewport on the 1920x1080 fullscreen
    layout. OCR is therefore restricted to that viewport instead of the entire
    browser frame. The worker continuously OCRs the newest submitted viewport
    and drops stale queued frames; the main vision loop consumes the newest
    completed result.
    """

    # Fullscreen 1920x1080 game viewport, measured from the actual game layout.
    # Keep a small margin around the visible card so floating text near its edge
    # is still captured. Coordinates are screen-space (left, top, width, height).
    _GAME_ROI = (540, 75, 575, 815)
    # OCR is deliberately sampled rather than submitted on every vision frame.
    # This keeps Tesseract from monopolizing CPU while its latest result remains
    # available asynchronously to the fast vision loop.
    _OCR_INTERVAL_FRAMES = 2

    _MIN_CONFIDENCE = 35.0
    _OVERLAP_REJECT = 0.18
    _PADDING_PX = 18
    _MAX_AGE_FRAMES = 20

    _UI_WORDS = {
        "score", "points", "point", "combo", "level", "time", "timer",
        "best", "high", "game", "start", "pause", "play", "menu",
        "round", "stage", "lives", "life",
    }

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._condition = threading.Condition(self._lock)
        self._pending: tuple[np.ndarray, int, int, int] | None = None
        self._latest_words: tuple[OCRWord, ...] = ()
        self._latest_frame_id = -1
        self._worker = threading.Thread(
            target=self._worker_loop,
            name="balloon-ocr",
            daemon=True,
        )
        self._worker.start()

    def observe(self, frame: np.ndarray, frame_id: int) -> OCRSnapshot:
        with self._condition:
            if frame_id % self._OCR_INTERVAL_FRAMES == 0:
                roi, origin_x, origin_y = self._crop_game_roi(frame)
                # Only copy the small game viewport. Copying the full 1920x1080
                # frame here would erase much of the benefit of ROI OCR.
                self._pending = (roi.copy(), frame_id, origin_x, origin_y)
                self._condition.notify()
            age = frame_id - self._latest_frame_id if self._latest_frame_id >= 0 else frame_id + 1
            return OCRSnapshot(
                words=self._latest_words,
                completed_frame_id=self._latest_frame_id,
                age_frames=max(0, age),
            )

    @classmethod
    def _crop_game_roi(cls, frame: np.ndarray) -> tuple[np.ndarray, int, int]:
        """Return the game viewport plus its screen-space origin.

        The configured ROI is for the 1920x1080 fullscreen layout. If a smaller
        or differently shaped frame is supplied, clamp it rather than failing;
        this keeps unit tests and non-fullscreen callers safe.
        """
        frame_height, frame_width = frame.shape[:2]
        x, y, width, height = cls._GAME_ROI
        x0 = max(0, min(x, frame_width))
        y0 = max(0, min(y, frame_height))
        x1 = max(x0, min(x + width, frame_width))
        y1 = max(y0, min(y + height, frame_height))
        if x1 <= x0 or y1 <= y0:
            return frame, 0, 0
        return frame[y0:y1, x0:x1], x0, y0

    def classify_bbox(
        self,
        bbox: tuple[int, int, int, int],
        *,
        snapshot: OCRSnapshot | None = None,
    ) -> dict[str, Any]:
        if snapshot is None:
            with self._lock:
                latest_frame_id = self._latest_frame_id
                words = self._latest_words
                age_frames = 0
        else:
            latest_frame_id = snapshot.completed_frame_id
            words = snapshot.words
            age_frames = snapshot.age_frames

        if latest_frame_id < 0:
            return {
                "ocr_text": [],
                "ocr_confidence": 0.0,
                "ocr_numeric": False,
                "ocr_ui_text": False,
                "ocr_twin_text": False,
                "ocr_overlap": 0.0,
                "ocr_age_frames": age_frames,
            }

        x, y, w, h = bbox
        candidate_area = max(1, w * h)
        best_overlap = 0.0
        best_confidence = 0.0
        matched: list[str] = []
        numeric = False
        ui_text = False
        twin_text = False

        for word in words:
            if word.confidence < self._MIN_CONFIDENCE:
                continue
            wx, wy, ww, wh = word.bbox
            px = self._PADDING_PX
            left, top = wx - px, wy - px
            right, bottom = wx + ww + px, wy + wh + px

            ix0 = max(x, left)
            iy0 = max(y, top)
            ix1 = min(x + w, right)
            iy1 = min(y + h, bottom)
            if ix1 <= ix0 or iy1 <= iy0:
                continue

            intersection = (ix1 - ix0) * (iy1 - iy0)
            overlap = intersection / float(candidate_area)
            center_inside = (
                left <= x + w / 2.0 <= right
                and top <= y + h / 2.0 <= bottom
            )
            if overlap < self._OVERLAP_REJECT and not center_inside:
                continue

            best_overlap = max(best_overlap, overlap)
            best_confidence = max(best_confidence, word.confidence)
            matched.append(word.text)

            if word.is_numeric:
                numeric = True
            if word.normalized in self._UI_WORDS:
                ui_text = True
            if word.is_twin_text:
                twin_text = True

        stale = age_frames > self._MAX_AGE_FRAMES
        return {
            "ocr_text": matched,
            "ocr_confidence": best_confidence,
            "ocr_numeric": numeric and not stale,
            "ocr_ui_text": ui_text and not stale,
            "ocr_twin_text": twin_text and not stale,
            "ocr_overlap": best_overlap,
            "ocr_age_frames": age_frames,
        }

    def _worker_loop(self) -> None:
        while True:
            with self._condition:
                while self._pending is None:
                    self._condition.wait()
                frame, frame_id, origin_x, origin_y = self._pending
                self._pending = None

            try:
                words = self._run_ocr(frame, origin=(origin_x, origin_y))
            except Exception:
                words = ()

            with self._lock:
                self._latest_words = tuple(words)
                self._latest_frame_id = frame_id

    @classmethod
    def _run_ocr(
        cls,
        frame: np.ndarray,
        *,
        origin: tuple[int, int] = (0, 0),
    ) -> tuple[OCRWord, ...]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        data = pytesseract.image_to_data(
            gray,
            config="--oem 3 --psm 11",
            output_type=pytesseract.Output.DICT,
        )

        words: list[OCRWord] = []
        count = len(data.get("text", []))
        for index in range(count):
            text = str(data["text"][index]).strip()
            if not text:
                continue
            try:
                confidence = float(data["conf"][index])
            except (TypeError, ValueError):
                continue
            if confidence < cls._MIN_CONFIDENCE:
                continue

            x = int(data["left"][index]) + origin[0]
            y = int(data["top"][index]) + origin[1]
            w = int(data["width"][index])
            h = int(data["height"][index])
            if w <= 0 or h <= 0:
                continue
            words.append(OCRWord(text, confidence, (x, y, w, h)))
        return tuple(words)
