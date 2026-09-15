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

    def grab(self) -> tuple[np.ndarray, float]:
        raw = self._sct.grab(self._monitor)
        arr = np.asarray(raw)[:, :, :3]
        ts = time.perf_counter_ns() / 1e9
        return arr, ts

    def close(self) -> None:
        self._sct.close()
