from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ReactionTimeConfig:
    distribution: str = "lognormal"
    median_ms: float = 300.0
    sigma: float = 0.45
    minimum_ms: float = 150.0
    maximum_ms: float = 2500.0


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
    enabled: bool = False
    miss_probability: float = 0.005
    delayed_response_probability: float = 0.02
    extra_pause_probability: float = 0.01
    extra_pause_min_ms: float = 200.0
    extra_pause_max_ms: float = 800.0


@dataclass
class WarmupConfig:
    enabled: bool = True
    duration_s: float = 30.0
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


@dataclass
class BehaviorConfig:
    enabled: bool = False
    seed: int = 12345
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

    @classmethod
    def from_dict(cls, data: dict) -> BehaviorConfig:
        rt_data = data.get("reaction_time", {})
        mv_data = data.get("movement", {})
        tr_data = data.get("trajectory", {})
        jt_data = data.get("jitter", {})
        ck_data = data.get("click", {})
        er_data = data.get("errors", {})
        wu_data = data.get("warmup", {})
        fg_data = data.get("fatigue", {})
        mp_data = data.get("micro_pauses", {})
        ep_data = data.get("endpoint_error", {})

        return cls(
            enabled=data.get("enabled", False),
            seed=data.get("seed", 12345),
            reaction_time=ReactionTimeConfig(**rt_data) if rt_data else ReactionTimeConfig(),
            movement=MovementConfig(**mv_data) if mv_data else MovementConfig(),
            trajectory=TrajectoryConfig(**tr_data) if tr_data else TrajectoryConfig(),
            jitter=JitterConfig(**jt_data) if jt_data else JitterConfig(),
            click=ClickConfig(**ck_data) if ck_data else ClickConfig(),
            errors=ErrorConfig(**er_data) if er_data else ErrorConfig(),
            warmup=WarmupConfig(**wu_data) if wu_data else WarmupConfig(),
            fatigue=FatigueConfig(**fg_data) if fg_data else FatigueConfig(),
            micro_pauses=MicroPauseConfig(**mp_data) if mp_data else MicroPauseConfig(),
            endpoint_error=EndpointErrorConfig(**ep_data) if ep_data else EndpointErrorConfig(),
        )
