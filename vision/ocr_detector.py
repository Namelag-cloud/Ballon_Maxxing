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
    """Asynchronous full-screen OCR used as a safety/semantic layer.

    The worker continuously OCRs the newest submitted frame and drops stale
    queued frames. The main vision loop consumes the newest completed result.
    """

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
        self._pending: tuple[np.ndarray, int] | None = None
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
            self._pending = (frame.copy(), frame_id)
            self._condition.notify()
            age = frame_id - self._latest_frame_id if self._latest_frame_id >= 0 else frame_id + 1
            return OCRSnapshot(
                words=self._latest_words,
                completed_frame_id=self._latest_frame_id,
                age_frames=max(0, age),
            )

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
                frame, frame_id = self._pending
                self._pending = None

            try:
                words = self._run_ocr(frame)
            except Exception:
                words = ()

            with self._lock:
                self._latest_words = tuple(words)
                self._latest_frame_id = frame_id

    @classmethod
    def _run_ocr(cls, frame: np.ndarray) -> tuple[OCRWord, ...]:
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

            x = int(data["left"][index])
            y = int(data["top"][index])
            w = int(data["width"][index])
            h = int(data["height"][index])
            if w <= 0 or h <= 0:
                continue
            words.append(OCRWord(text, confidence, (x, y, w, h)))
        return tuple(words)
