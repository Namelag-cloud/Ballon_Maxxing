from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Optional


@dataclass
class SessionOutcome:
    """Bot-side observations and estimates; never authoritative engine state."""

    target_id: Optional[int] = None
    target_type: Optional[str] = None
    target_radius_px: Optional[float] = None
    target_first_seen_ts: Optional[float] = None
    acquisition_ts: Optional[float] = None
    planned_dispatch_ts: Optional[float] = None
    behavioral_planning_ts: Optional[float] = None
    dispatch_ts: Optional[float] = None
    actual_click_down_ts: Optional[float] = None
    observed_pop_ts: Optional[float] = None
    estimated_reaction_ticks: Optional[float] = None
    estimated_acquisition_click_ticks: Optional[float] = None
    estimated_click_tick: Optional[float] = None
    click_tick_modulo: Optional[int] = None
    timing_error_ms: Optional[float] = None
    intended_vs_observed_timing_error_ms: Optional[float] = None
    modeled_perception_ticks: Optional[float] = None
    scheduler_delay_ms: Optional[float] = None
    movement_duration_ms: Optional[float] = None
    screen_click_x: Optional[int] = None
    screen_click_y: Optional[int] = None
    world_x_estimate: Optional[float] = None
    world_y_estimate: Optional[float] = None
    endpoint_error_px: Optional[float] = None
    intended_reaction_ticks: Optional[float] = None
    intended_reaction_s: Optional[float] = None
    hit: Optional[bool] = None
    notification: Optional[bool] = None
    gemini_observed: Optional[bool] = None
    gemini_selected: Optional[bool] = None
    gemini_executed: Optional[bool] = None
    gemini_completed: Optional[bool] = None
    duo_delta_ms: Optional[float] = None
    special_sequence_outcome: Optional[str] = None
    balloon_g_outcome: Optional[str] = None
    feedback_confidence: float = 0.0
    observation_notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class TimingCalibration:
    intended_reaction_ticks: list[float] = field(default_factory=list)
    observed_reaction_ticks: list[float] = field(default_factory=list)
    wall_clock_error_ms: list[float] = field(default_factory=list)
    click_tick_modulo: list[int] = field(default_factory=list)

    def record(self, *, intended_ticks: float, observed_ticks: Optional[float], wall_clock_error_ms: float) -> None:
        self.intended_reaction_ticks.append(float(intended_ticks))
        if observed_ticks is not None:
            self.observed_reaction_ticks.append(float(observed_ticks))
        self.wall_clock_error_ms.append(float(wall_clock_error_ms))
        if observed_ticks is not None:
            self.click_tick_modulo.append(int(round(observed_ticks)) % 3)

    def summary(self) -> dict:
        def stats(values):
            if not values:
                return {"count": 0}
            ordered = sorted(values)
            n = len(ordered)
            median = ordered[n // 2] if n % 2 else (ordered[n // 2 - 1] + ordered[n // 2]) / 2
            return {
                "count": n,
                "mean": sum(values) / n,
                "median": median,
                "min": min(values),
                "max": max(values),
            }

        return {
            "intended_reaction_ticks": stats(self.intended_reaction_ticks),
            "observed_reaction_ticks": stats(self.observed_reaction_ticks),
            "wall_clock_error_ms": stats(self.wall_clock_error_ms),
            "click_tick_modulo": stats(self.click_tick_modulo),
        }
