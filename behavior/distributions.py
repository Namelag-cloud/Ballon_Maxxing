from __future__ import annotations

import math
from typing import Optional

import numpy as np


class SeededRNG:
    def __init__(self, seed: int = 12345) -> None:
        self._seed = seed
        self._rng = np.random.RandomState(seed)

    @property
    def seed(self) -> int:
        return self._seed

    def random(self) -> float:
        return float(self._rng.random())

    def normal(self, loc: float = 0.0, scale: float = 1.0) -> float:
        return float(self._rng.normal(loc, scale))

    def uniform(self, low: float = 0.0, high: float = 1.0) -> float:
        return float(self._rng.uniform(low, high))

    def lognormal(self, mean: float = 0.0, sigma: float = 1.0) -> float:
        return float(self._rng.lognormal(mean, sigma))

    def choice(self, arr, size=None, replace=True):
        return self._rng.choice(arr, size=size, replace=replace)

    def rayleigh(self, scale: float = 1.0) -> float:
        return float(self._rng.rayleigh(scale))

    def exponential(self, scale: float = 1.0) -> float:
        return float(self._rng.exponential(scale))

    def beta(self, a: float, b: float) -> float:
        return float(self._rng.beta(a, b))


def sample_lognormal_bounded(
    rng: SeededRNG,
    median_ms: float,
    sigma: float,
    minimum_ms: float,
    maximum_ms: float,
) -> float:
    if median_ms <= 0:
        raise ValueError("median_ms must be positive")
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    if minimum_ms < 0:
        raise ValueError("minimum_ms must be non-negative")
    if maximum_ms < minimum_ms:
        raise ValueError("maximum_ms must be >= minimum_ms")

    log_median = math.log(median_ms)
    raw = rng.lognormal(log_median, sigma)

    return max(minimum_ms, min(maximum_ms, raw))


def sample_fitts_mt(
    rng: SeededRNG,
    distance: float,
    width: float,
    a_ms: float,
    b_ms: float,
    variance: float,
) -> float:
    if distance <= 0:
        return a_ms

    w = max(width, 1.0)
    id_value = math.log2(1.0 + distance / w)
    predicted = a_ms + b_ms * id_value

    if variance > 0:
        log_var = math.log(1.0 + variance)
        sampled = rng.lognormal(math.log(predicted), log_var)
    else:
        sampled = predicted

    return max(1.0, sampled)
