from __future__ import annotations

import time

import mss
import numpy as np

from capture.base import CaptureBackend
from config.settings import CaptureConfig


class MSSCapture(CaptureBackend):
    def __init__(self, config: CaptureConfig) -> None:
        self._config = config
        r = config.region
        self._monitor = {
            "top": r.top,
            "left": r.left,
            "width": r.width,
            "height": r.height,
        }
        self._sct = mss.MSS()
        self._last_timing_ms = {"capture": 0.0, "numpy": 0.0}

    @property
    def last_timing_ms(self) -> dict[str, float]:
        return dict(self._last_timing_ms)

    def grab(self) -> tuple[np.ndarray, float]:
        t0 = time.perf_counter_ns()
        raw = self._sct.grab(self._monitor)
        t1 = time.perf_counter_ns()
        arr = np.asarray(raw)[:, :, :3]
        t2 = time.perf_counter_ns()
        self._last_timing_ms = {
            "capture": (t1 - t0) / 1e6,
            "numpy": (t2 - t1) / 1e6,
        }
        ts = t2 / 1e9
        return arr, ts

    def close(self) -> None:
        self._sct.close()
