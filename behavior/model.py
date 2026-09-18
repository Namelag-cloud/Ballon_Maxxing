from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass
from typing import Optional

from behavior.config import BehaviorConfig
from behavior.distributions import SeededRNG, sample_fitts_mt
from behavior.movement import (
    TrajectoryPoint,
    compute_fitts_id,
    generate_trajectory,
    apply_jitter,
)
from behavior.profile import HumanProfile
from behavior.scheduler import SessionScheduler, SessionState


ENGINE_TICK_MS = 1000.0 / 60.0


@dataclass
class BehavioralAction:
    reaction_time_s: float
    movement_time_s: float
    click_duration_s: float
    trajectory: list[TrajectoryPoint]
    total_time_s: float
    start_position: tuple[float, float]
    target_position: tuple[float, float]
    click_position: tuple[float, float]
    distance: float
    target_width: float
    fitts_id: float
    endpoint_error_px: float
    pause_duration_s: float
    is_miss: bool
    is_delayed: bool
    session_modifier: float
    warmup_factor: float
    fatigue_factor: float
    intended_reaction_ticks: float = 0.0
    perception_ticks: float = 0.0
    planned_at: float = 0.0
    acquisition_timestamp: float | None = None
    scheduler_delay_s: float = 0.0


class BehavioralModel:
    """Stochastic session model; the RNG is bot-only and never reads game state."""

    def __init__(self, config: BehaviorConfig) -> None:
        seed = config.seed
        if seed is None:
            seed = int.from_bytes(os.urandom(8), "little", signed=False) & 0x7FFFFFFF
        self._rng = SeededRNG(seed)
        self._config = config
        self._profile = HumanProfile.sample(self._rng, {
            "reaction_sigma": config.human_profile.reaction_sigma,
            "reaction_ar1": config.human_profile.reaction_ar1,
            "perception_min": config.human_profile.perception_min_ticks,
            "perception_max": config.human_profile.perception_max_ticks,
            "spatial_sigma_pct_min": config.human_profile.spatial_sigma_pct_min,
            "spatial_sigma_pct_max": config.human_profile.spatial_sigma_pct_max,
            "miss_rate_min": config.human_profile.miss_rate_min,
            "miss_rate_max": config.human_profile.miss_rate_max,
            "notif_hit_rate_min": config.human_profile.notif_hit_rate_min,
            "notif_hit_rate_max": config.human_profile.notif_hit_rate_max,
            "duo_delta_min_ms": config.human_profile.duo_delta_min_ms,
            "duo_delta_max_ms": config.human_profile.duo_delta_max_ms,
            "super_g_fail_rate_min": config.human_profile.super_g_fail_rate_min,
            "super_g_fail_rate_max": config.human_profile.super_g_fail_rate_max,
            "balloon_g_abandon_rate_min": config.human_profile.balloon_g_abandon_rate_min,
            "balloon_g_abandon_rate_max": config.human_profile.balloon_g_abandon_rate_max,
            "idle_lambda_min": config.human_profile.idle_lambda_min,
            "idle_lambda_max": config.human_profile.idle_lambda_max,
            "start_delay_min_s": config.human_profile.start_delay_min_s,
            "start_delay_max_s": config.human_profile.start_delay_max_s,
            "warmup_min_s": config.human_profile.warmup_min_s,
            "warmup_max_s": config.human_profile.warmup_max_s,
            "fatigue_min_ticks_per_min": config.human_profile.fatigue_min_ticks_per_min,
            "fatigue_max_ticks_per_min": config.human_profile.fatigue_max_ticks_per_min,
            "early_exit_prob_min": config.human_profile.early_exit_prob_min,
            "early_exit_prob_max": config.human_profile.early_exit_prob_max,
            "early_exit_ratio_min": config.human_profile.early_exit_ratio_min,
            "early_exit_ratio_max": config.human_profile.early_exit_ratio_max,
        })
        self._session = SessionScheduler(
            self._rng,
            config.warmup,
            config.fatigue,
            config.micro_pauses,
        )
        self._session_started = False
        self._session_start = 0.0
        self._last_reaction_ticks = self._profile.reaction_mu_ticks
        self._sampled_start_delay_s = self._profile.start_delay_sec
        self._early_exit_selected = False
        self._activity_cutoff_s = 1.5

    @property
    def config(self) -> BehaviorConfig:
        return self._config

    @property
    def rng(self) -> SeededRNG:
        return self._rng

    @property
    def profile(self) -> HumanProfile:
        return self._profile

    @property
    def behavioral_seed(self) -> int:
        return self._rng.seed

    @property
    def start_delay_s(self) -> float:
        return self._sampled_start_delay_s

    def start_session(self) -> None:
        self._session.start_session()
        self._session_started = True
        self._session_start = time.perf_counter()
        self._early_exit_selected = self._rng.random() < self._profile.early_exit_prob

    def session_elapsed_s(self) -> float:
        return max(0.0, time.perf_counter() - self._session_start) if self._session_started else 0.0

    def should_early_exit(self, duration_s: float | None) -> bool:
        if duration_s is None or duration_s <= 0 or not self._session_started:
            return False
        return self._early_exit_selected and self.session_elapsed_s() >= duration_s * self._profile.early_exit_ratio

    def should_start_new_action(self, duration_s: float | None) -> bool:
        if duration_s is None or duration_s <= 0 or not self._session_started:
            return True
        return self.session_elapsed_s() < max(0.0, duration_s - self._activity_cutoff_s)

    def actions_ready(self) -> bool:
        return self.session_elapsed_s() >= self._sampled_start_delay_s

    def activity_cutoff_s(self) -> float:
        return self._activity_cutoff_s

    def should_click_notification(self) -> bool:
        return self._rng.random() < self._profile.notif_hit_rate

    def sample_idle_gap_s(self) -> float:
        if self._profile.idle_lambda <= 0:
            return 0.0
        # Exponential waiting time, with no forced idle on every action.
        if self._rng.random() > min(1.0, self._profile.idle_lambda):
            return 0.0
        return self._rng.exponential(1.0 / self._profile.idle_lambda)

    def sample_duo_delta_s(self) -> float:
        raw = self._rng.lognormal(math.log(self._profile.duo_delta_mu_ms), self._profile.duo_delta_sigma)
        ms = max(self._profile.duo_delta_min_ms, min(self._profile.duo_delta_max_ms, raw))
        return ms / 1000.0

    def plan_action(
        self,
        current_x: float,
        current_y: float,
        target_x: float,
        target_y: float,
        target_width: float,
        target_metadata: Optional[dict] = None,
        acquisition_timestamp: float | None = None,
    ) -> BehavioralAction:
        if not self._session_started:
            self.start_session()

        elapsed = self.session_elapsed_s()
        state = self._session.update()
        warmup_factor = max(0.0, self._profile.warmup_sec - elapsed) / max(0.001, self._profile.warmup_sec) * 0.15
        fatigue_factor = max(0.0, min(0.30, max(0.0, elapsed - 300.0) / 60.0 * self._profile.fatigue_per_min / 100.0))
        modifier = warmup_factor + fatigue_factor

        jitter_x, jitter_y = current_x, current_y
        if self._config.jitter.enabled:
            jitter_x, jitter_y = apply_jitter(
                current_x, current_y, self._rng,
                self._config.jitter.amplitude_px,
                self._config.jitter.smoothness,
            )

        is_miss = self._rng.random() < self._profile.miss_rate if self._config.errors.enabled else False
        is_delayed = self._rng.random() < self._config.errors.delayed_response_probability if self._config.errors.enabled else False

        if is_miss:
            radius = max(1.0, target_width / 2.0)
            distance = self._rng.uniform(
                self._config.errors.miss_min_radius,
                self._config.errors.miss_max_radius,
            ) * radius
            angle = self._rng.uniform(0.0, math.tau)
            actual_target_x = target_x + math.cos(angle) * distance
            actual_target_y = target_y + math.sin(angle) * distance
        else:
            sigma = max(
                0.1,
                self._profile.spatial_sigma_pct * max(1.0, target_width / 2.0),
            )
            if self._config.endpoint_error.enabled:
                sigma = max(sigma, self._config.endpoint_error.scale_px)
                sigma = min(sigma, max(1.0, self._config.endpoint_error.max_px))
            dx = self._rng.normal(0.0, sigma)
            dy = self._rng.normal(0.0, sigma)
            # Small correlated hand/fat-finger component.
            fat = self._rng.normal(0.0, self._profile.fat_finger_sigma_pct * max(1.0, target_width / 2.0))
            actual_target_x = target_x + dx + fat
            actual_target_y = target_y + dy + fat

        distance = math.hypot(target_x - current_x, target_y - current_y)

        base = self._profile.reaction_mu_ticks
        sample = self._rng.lognormal(math.log(max(1.0, base)), self._profile.reaction_sigma)
        correlated = self._profile.reaction_ar1 * self._last_reaction_ticks + (1.0 - self._profile.reaction_ar1) * sample
        intended_ticks = max(9.0, correlated)
        if is_delayed:
            intended_ticks += self._rng.uniform(12.0, 48.0)
        intended_ticks *= (1.0 + modifier)
        self._last_reaction_ticks = intended_ticks
        # Perception is a modeled component of the intended acquisition-to-click
        # interval. Do not subtract it from the scheduled delay: doing so makes
        # the observed interval systematically shorter than the intended one.
        reaction_s = intended_ticks * ENGINE_TICK_MS / 1000.0

        raw_movement = sample_fitts_mt(
            self._rng, distance, max(1.0, target_width),
            self._config.movement.a_ms, self._config.movement.b_ms,
            self._config.movement.variance,
        )
        movement_s = raw_movement * (1.0 + modifier) / 1000.0

        raw_click = self._rng.lognormal(
            math.log(self._config.click.median_ms),
            self._config.click.sigma,
        )
        click_s = max(
            self._config.click.minimum_ms,
            min(self._config.click.maximum_ms, raw_click),
        ) / 1000.0

        pause_s = 0.0
        micro_pause_ms = self._session.should_micro_pause()
        if micro_pause_ms is not None:
            pause_s += micro_pause_ms / 1000.0

        trajectory: list[TrajectoryPoint] = []
        if self._config.trajectory.enabled and movement_s > 0.001 and distance > 1.0:
            trajectory = generate_trajectory(
                jitter_x, jitter_y, actual_target_x, actual_target_y, movement_s,
                self._rng,
                num_points=self._config.trajectory.points,
                lateral_deviation_px=self._config.trajectory.lateral_deviation_px,
                velocity_variance=self._config.trajectory.velocity_variance,
                endpoint_correction=self._config.trajectory.endpoint_correction,
            )

        endpoint_err = math.hypot(actual_target_x - target_x, actual_target_y - target_y)
        total_s = reaction_s + movement_s + click_s + pause_s
        planned_at = time.perf_counter()

        return BehavioralAction(
            reaction_time_s=reaction_s,
            movement_time_s=movement_s,
            click_duration_s=click_s,
            trajectory=trajectory,
            total_time_s=total_s,
            start_position=(jitter_x, jitter_y),
            target_position=(actual_target_x, actual_target_y),
            click_position=(actual_target_x, actual_target_y),
            distance=distance,
            target_width=target_width,
            fitts_id=compute_fitts_id(distance, max(1.0, target_width)),
            endpoint_error_px=endpoint_err,
            pause_duration_s=pause_s,
            is_miss=is_miss,
            is_delayed=is_delayed,
            session_modifier=modifier,
            warmup_factor=warmup_factor,
            fatigue_factor=fatigue_factor,
            intended_reaction_ticks=intended_ticks,
            perception_ticks=self._profile.perception_ticks,
            planned_at=planned_at,
            acquisition_timestamp=acquisition_timestamp,
            scheduler_delay_s=0.0,
        )

    def wait_reaction(self, action: BehavioralAction) -> None:
        if action.reaction_time_s > 0:
            time.sleep(action.reaction_time_s)

    def execute_trajectory(self, action: BehavioralAction, move_fn, sleep_fn=None) -> None:
        if not action.trajectory:
            return
        sleep_fn = sleep_fn or time.sleep
        prev_t = 0.0
        for pt in action.trajectory:
            dt = pt.t - prev_t
            if dt > 0:
                sleep_fn(dt)
            move_fn(int(pt.x), int(pt.y))
            prev_t = pt.t

    def wait_click_duration(self, action: BehavioralAction) -> None:
        if action.click_duration_s > 0:
            time.sleep(action.click_duration_s)

    def wait_pause(self, action: BehavioralAction) -> None:
        if action.pause_duration_s > 0:
            time.sleep(action.pause_duration_s)

    def get_session_state(self) -> SessionState:
        return self._session.update()
