from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from behavior.distributions import SeededRNG


@dataclass(frozen=True)
class HumanProfile:
    """Per-session behavioral parameters.

    Defaults are provisional model starting points, not measurements of a human
    population. Keep the sampled values in telemetry so a run is reproducible.
    """

    reaction_mu_ticks: float = 20.0
    reaction_sigma: float = 0.35
    perception_ticks: float = 3.5
    spatial_sigma_pct: float = 0.16
    spatial_sigma_px: float = 5.0
    miss_rate: float = 0.035
    notif_hit_rate: float = 0.01
    duo_delta_mu_ms: float = 150.0
    duo_delta_sigma: float = 0.4
    duo_delta_min_ms: float = 50.0
    duo_delta_max_ms: float = 500.0
    duo_y_variance_multiplier: float = 1.5
    super_g_fail_rate: float = 0.15
    balloon_g_abandon_rate: float = 0.10
    idle_lambda: float = 0.05
    start_delay_sec: float = 2.0
    session_duration_s: float = 60.0
    warmup_sec: float = 20.0
    fatigue_per_min: float = 1.0
    early_exit_prob: float = 0.10
    early_exit_ratio: float = 0.90
    reaction_ar1: float = 0.30
    fat_finger_probability: float = 0.02
    fat_finger_multiplier: float = 2.0
    gemini_without_notification_probability: float = 0.01
    final_activity_cutoff_s: float = 1.5

    @classmethod
    def sample(cls, rng: SeededRNG, data: dict[str, Any] | None = None) -> "HumanProfile":
        cfg = data or {}

        def uniform(name: str, low: float, high: float) -> float:
            return rng.uniform(float(cfg.get(f"{name}_min", low)), float(cfg.get(f"{name}_max", high)))

        def positive_lognormal(name: str, median: float, sigma: float, low: float, high: float) -> float:
            med = float(cfg.get(f"{name}_median", median))
            sig = float(cfg.get(f"{name}_sigma", sigma))
            return max(float(cfg.get(f"{name}_min", low)), min(float(cfg.get(f"{name}_max", high)), rng.lognormal(math.log(med), sig)))

        def clipped_normal(mean: float, sigma: float, low: float, high: float) -> float:
            return max(low, min(high, rng.normal(mean, sigma)))

        def beta_bounded(name: str, mean: float, low: float, high: float, concentration: float = 12.0) -> float:
            low = float(cfg.get(f"{name}_min", low))
            high = float(cfg.get(f"{name}_max", high))
            if high <= low:
                return low
            normalized_mean = (mean - low) / (high - low)
            alpha = max(0.1, normalized_mean * concentration)
            beta = max(0.1, (1.0 - normalized_mean) * concentration)
            return low + rng.beta(alpha, beta) * (high - low)

        return cls(
            # These are sampled once per session from meta-distributions.
            reaction_mu_ticks=clipped_normal(18.0, 2.0, 10.0, 30.0),
            reaction_sigma=rng.uniform(0.30, 0.40),
            perception_ticks=uniform("perception", 2.0, 5.0),
            spatial_sigma_pct=uniform("spatial_sigma_pct", 0.10, 0.22),
            spatial_sigma_px=clipped_normal(5.0, 1.5, 1.0, 10.0),
            miss_rate=beta_bounded("miss_rate", 0.02, 0.005, 0.05),
            notif_hit_rate=beta_bounded("notif_hit_rate", 0.005, 0.0, 0.02),
            duo_delta_mu_ms=positive_lognormal("duo_delta", 150.0, 0.25, 75.0, 300.0),
            duo_delta_sigma=float(cfg.get("duo_delta_sigma", 0.4)),
            duo_delta_min_ms=float(cfg.get("duo_delta_min_ms", 50.0)),
            duo_delta_max_ms=float(cfg.get("duo_delta_max_ms", 500.0)),
            duo_y_variance_multiplier=float(cfg.get("duo_y_variance_multiplier", 1.5)),
            super_g_fail_rate=uniform("super_g_fail_rate", 0.08, 0.22),
            balloon_g_abandon_rate=uniform("balloon_g_abandon_rate", 0.05, 0.15),
            idle_lambda=uniform("idle_lambda", 0.03, 0.08),
            start_delay_sec=uniform("start_delay_sec", 1.0, 3.0),
            session_duration_s=clipped_normal(60.0, 5.0, 50.0, 75.0),
            warmup_sec=uniform("warmup_sec", 15.0, 25.0),
            fatigue_per_min=uniform("fatigue_per_min", 0.5, 2.0),
            early_exit_prob=uniform("early_exit_prob", 0.05, 0.15),
            early_exit_ratio=uniform("early_exit_ratio", 0.80, 0.95),
            reaction_ar1=float(cfg.get("reaction_ar1", 0.30)),
            fat_finger_probability=float(cfg.get("fat_finger_probability", 0.02)),
            fat_finger_multiplier=float(cfg.get("fat_finger_multiplier", 2.0)),
            gemini_without_notification_probability=float(cfg.get("gemini_without_notification_probability", 0.01)),
            final_activity_cutoff_s=float(cfg.get("final_activity_cutoff_s", 1.5)),
        )

    def as_dict(self) -> dict[str, float]:
        return self.__dict__.copy()
