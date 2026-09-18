from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ReactionTimeConfig:
    distribution: str = "lognormal"
    median_ms: float = 333.32
    sigma: float = 0.35
    minimum_ms: float = 149.994
    maximum_ms: float = 2500.0
    perception_min_ticks: float = 2.0
    perception_max_ticks: float = 5.0
    ar1: float = 0.30


@dataclass
class MovementConfig:
    model: str = "fitts"
    a_ms: float = 80.0
    b_ms: float = 70.0
    variance: float = 0.15


@dataclass
class TrajectoryConfig:
    enabled: bool = True
    lateral_deviation_px: float = 2.0
    velocity_variance: float = 0.15
    endpoint_correction: bool = True
    points: int = 20


@dataclass
class JitterConfig:
    enabled: bool = True
    amplitude_px: float = 1.5
    smoothness: float = 0.8


@dataclass
class ClickConfig:
    duration_distribution: str = "lognormal"
    median_ms: float = 80.0
    sigma: float = 0.25
    minimum_ms: float = 25.0
    maximum_ms: float = 250.0


@dataclass
class ErrorConfig:
    enabled: bool = True
    miss_probability: float = 0.035
    delayed_response_probability: float = 0.02
    extra_pause_probability: float = 0.01
    extra_pause_min_ms: float = 200.0
    extra_pause_max_ms: float = 800.0
    miss_min_radius: float = 1.5
    miss_max_radius: float = 2.5


@dataclass
class WarmupConfig:
    enabled: bool = True
    duration_s: float = 20.0
    modifier: float = 0.15


@dataclass
class FatigueConfig:
    enabled: bool = True
    start_s: float = 300.0
    rate: float = 0.05
    maximum: float = 0.30


@dataclass
class MicroPauseConfig:
    enabled: bool = True
    probability: float = 0.01
    minimum_ms: float = 300.0
    maximum_ms: float = 1500.0


@dataclass
class EndpointErrorConfig:
    enabled: bool = True
    scale_px: float = 5.0
    max_px: float = 12.0
    spatial_sigma_pct: float = 0.16


@dataclass
class HumanProfileConfig:
    enabled: bool = True
    reaction_mu_ticks_min: float = 12.0
    reaction_mu_ticks_max: float = 32.0
    spatial_sigma_pct_min: float = 0.10
    spatial_sigma_pct_max: float = 0.22
    miss_rate_min: float = 0.02
    miss_rate_max: float = 0.05
    notif_hit_rate_min: float = 0.005
    notif_hit_rate_max: float = 0.02
    duo_delta_min_ms: float = 50.0
    duo_delta_max_ms: float = 500.0
    duo_y_variance_multiplier: float = 1.5
    duo_delta_median_ms: float = 150.0
    super_g_fail_rate_min: float = 0.08
    super_g_fail_rate_max: float = 0.22
    balloon_g_abandon_rate_min: float = 0.05
    balloon_g_abandon_rate_max: float = 0.15
    idle_lambda_min: float = 0.03
    idle_lambda_max: float = 0.08
    start_delay_min_s: float = 1.0
    start_delay_max_s: float = 3.0
    warmup_min_s: float = 15.0
    warmup_max_s: float = 25.0
    fatigue_min_ticks_per_min: float = 0.5
    fatigue_max_ticks_per_min: float = 2.0
    early_exit_prob_min: float = 0.05
    early_exit_prob_max: float = 0.15
    early_exit_ratio_min: float = 0.80
    early_exit_ratio_max: float = 0.95
    reaction_sigma: float = 0.35
    reaction_ar1: float = 0.30
    perception_min_ticks: float = 2.0
    perception_max_ticks: float = 5.0
    duo_delta_sigma: float = 0.4
    fat_finger_probability: float = 0.02
    fat_finger_multiplier: float = 2.0
    gemini_without_notification_probability: float = 0.01
    final_activity_cutoff_s: float = 1.5


@dataclass
class BehaviorConfig:
    enabled: bool = False
    # None means entropy-seeded per session. An explicit value is retained as
    # a reproducible debug override and never represents the game seed.
    seed: int | None = None
    reaction_time: ReactionTimeConfig = field(default_factory=ReactionTimeConfig)
    movement: MovementConfig = field(default_factory=MovementConfig)
    trajectory: TrajectoryConfig = field(default_factory=TrajectoryConfig)
    jitter: JitterConfig = field(default_factory=JitterConfig)
    click: ClickConfig = field(default_factory=ClickConfig)
    errors: ErrorConfig = field(default_factory=ErrorConfig)
    warmup: WarmupConfig = field(default_factory=WarmupConfig)
    fatigue: FatigueConfig = field(default_factory=FatigueConfig)
    micro_pauses: MicroPauseConfig = field(default_factory=MicroPauseConfig)
    endpoint_error: EndpointErrorConfig = field(default_factory=EndpointErrorConfig)
    human_profile: HumanProfileConfig = field(default_factory=HumanProfileConfig)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BehaviorConfig":
        def build(typ, key):
            value = data.get(key, {})
            return typ(**value) if value else typ()

        seed = data.get("seed")
        return cls(
            enabled=data.get("enabled", False),
            seed=int(seed) if seed is not None else None,
            reaction_time=build(ReactionTimeConfig, "reaction_time"),
            movement=build(MovementConfig, "movement"),
            trajectory=build(TrajectoryConfig, "trajectory"),
            jitter=build(JitterConfig, "jitter"),
            click=build(ClickConfig, "click"),
            errors=build(ErrorConfig, "errors"),
            warmup=build(WarmupConfig, "warmup"),
            fatigue=build(FatigueConfig, "fatigue"),
            micro_pauses=build(MicroPauseConfig, "micro_pauses"),
            endpoint_error=build(EndpointErrorConfig, "endpoint_error"),
            human_profile=build(HumanProfileConfig, "human_profile"),
        )
