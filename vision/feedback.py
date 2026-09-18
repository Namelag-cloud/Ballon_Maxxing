from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np


@dataclass(frozen=True)
class FeedbackObservation:
    timestamp: float
    combo_present: bool
    combo_value: Optional[int]
    pop_detected: bool
    score_changed: bool
    score_value: Optional[int]
    confidence: float
    notes: tuple[str, ...] = ()


class FeedbackReader:
    """Screen-only feedback reader.

    The game UI layout is not part of the frozen source available to the bot,
    so no fixed ROI or OCR oracle is assumed. Signals remain UNOBSERVABLE until
    the caller supplies a validated ROI/template configuration. A generic
    frame-change detector is available for calibration, but is deliberately
    not promoted to an authoritative hit/miss signal.
    """

    UNOBSERVABLE = "UNOBSERVABLE THROUGH CURRENT INTERFACE"

    def __init__(self, *, combo_roi=None, score_roi=None, pop_roi=None, change_threshold: float = 0.08):
        self.combo_roi = combo_roi
        self.score_roi = score_roi
        self.pop_roi = pop_roi
        self.change_threshold = float(change_threshold)
        self._previous = {"combo": None, "score": None, "pop": None}

    @staticmethod
    def _crop(frame: np.ndarray, roi):
        if roi is None:
            return None
        x, y, w, h = map(int, roi)
        if w <= 0 or h <= 0:
            return None
        return frame[max(0, y):max(0, y) + h, max(0, x):max(0, x) + w]

    def _change(self, current, key: str) -> bool:
        if current is None or current.size == 0:
            return False
        gray = cv2.cvtColor(current, cv2.COLOR_BGR2GRAY)
        previous = self._previous.get(key)
        self._previous[key] = gray.copy()
        if previous is None or previous.shape != gray.shape:
            return False
        delta = float(np.mean(cv2.absdiff(gray, previous))) / 255.0
        return delta >= self.change_threshold

    def observe(self, frame: np.ndarray, timestamp: float) -> FeedbackObservation:
        combo = self._crop(frame, self.combo_roi)
        score = self._crop(frame, self.score_roi)
        pop = self._crop(frame, self.pop_roi)
        combo_changed = self._change(combo, "combo")
        score_changed = self._change(score, "score")
        pop_changed = self._change(pop, "pop")

        notes = []
        if self.combo_roi is None:
            notes.append(f"combo: {self.UNOBSERVABLE}")
        if self.score_roi is None:
            notes.append(f"score: {self.UNOBSERVABLE}")
        if self.pop_roi is None:
            notes.append(f"pop: {self.UNOBSERVABLE}")

        configured = sum(v is not None for v in (self.combo_roi, self.score_roi, self.pop_roi))
        confidence = 0.0 if configured == 0 else min(1.0, configured / 3.0)

        return FeedbackObservation(
            timestamp=timestamp,
            combo_present=combo is not None,
            combo_value=None,
            pop_detected=pop_changed if pop is not None else False,
            score_changed=score_changed,
            score_value=None,
            confidence=confidence,
            notes=tuple(notes),
        )
