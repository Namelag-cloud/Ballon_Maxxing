from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

from behavior.distributions import SeededRNG
from behavior.config import WarmupConfig, FatigueConfig, MicroPauseConfig


@dataclass
class SessionState:
    elapsed_s: float = 0.0
    warmup_factor: float = 0.0
    fatigue_factor: float = 0.0
    modifier: float = 0.0


class SessionScheduler:
    def __init__(
        self,
        rng: SeededRNG,
        warmup: WarmupConfig,
        fatigue: FatigueConfig,
        micro_pauses: MicroPauseConfig,
    ) -> None:
        self._rng = rng
        self._warmup = warmup
        self._fatigue = fatigue
        self._micro_pauses = micro_pauses
        self._session_start: Optional[float] = None
        self._last_action_time: float = 0.0

    def start_session(self) -> None:
        self._session_start = time.perf_counter()

    def update(self) -> SessionState:
        if self._session_start is None:
            return SessionState()

        elapsed = time.perf_counter() - self._session_start
        warmup_f = self._compute_warmup(elapsed)
        fatigue_f = self._compute_fatigue(elapsed)
        modifier = warmup_f + fatigue_f

        return SessionState(
            elapsed_s=elapsed,
            warmup_factor=warmup_f,
            fatigue_factor=fatigue_f,
            modifier=modifier,
        )

    def _compute_warmup(self, elapsed: float) -> float:
        if not self._warmup.enabled or self._warmup.duration_s <= 0:
            return 0.0

        if elapsed >= self._warmup.duration_s:
            return 0.0

        progress = elapsed / self._warmup.duration_s
        return self._warmup.modifier * (1.0 - progress)

    def _compute_fatigue(self, elapsed: float) -> float:
        if not self._fatigue.enabled or elapsed < self._fatigue.start_s:
            return 0.0

        active_time = elapsed - self._fatigue.start_s
        fatigue = self._fatigue.rate * (active_time / 60.0)
        return min(fatigue, self._fatigue.maximum)

    def should_micro_pause(self) -> Optional[float]:
        if not self._micro_pauses.enabled:
            return None

        if self._rng.random() < self._micro_pauses.probability:
            return self._rng.uniform(
                self._micro_pauses.minimum_ms,
                self._micro_pauses.maximum_ms,
            )
        return None

    def apply_modifier(self, base_ms: float, state: SessionState) -> float:
        if state.modifier == 0:
            return base_ms
        return base_ms * (1.0 + state.modifier)

    def should_miss(self, base_probability: float, state: SessionState) -> bool:
        if base_probability <= 0:
            return False
        adjusted = base_probability * (1.0 + state.modifier * 2.0)
        return self._rng.random() < adjusted

    def should_delay(self, base_probability: float, state: SessionState) -> bool:
        if base_probability <= 0:
            return False
        adjusted = base_probability * (1.0 + state.modifier)
        return self._rng.random() < adjusted
