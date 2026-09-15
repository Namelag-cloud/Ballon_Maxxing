from __future__ import annotations

import time
from dataclasses import dataclass, field


class Timer:
    def __init__(self) -> None:
        self._start: float = 0.0
        self._laps: list[float] = []

    def start(self) -> None:
        self._start = time.perf_counter_ns() / 1e9
        self._laps.clear()

    def lap(self) -> float:
        now = time.perf_counter_ns() / 1e9
        self._laps.append(now)
        return now

    def elapsed_ns(self) -> float:
        return (time.perf_counter_ns() / 1e9 - self._start) * 1e9

    def elapsed_ms(self) -> float:
        return self.elapsed_ns() / 1e6

    @property
    def laps(self) -> list[float]:
        return list(self._laps)


def now_monotonic() -> float:
    return time.perf_counter_ns() / 1e9
