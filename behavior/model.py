from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Optional

from behavior.config import BehaviorConfig
from behavior.distributions import SeededRNG, sample_lognormal_bounded, sample_fitts_mt
from behavior.movement import (
    TrajectoryPoint,
    MovementPlan,
    compute_fitts_id,
    generate_trajectory,
    apply_jitter,
    compute_endpoint_error,
)
from behavior.scheduler import SessionScheduler, SessionState


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


class BehavioralModel:
    def __init__(self, config: BehaviorConfig) -> None:
        self._config = config
        self._rng = SeededRNG(config.seed)
        self._session = SessionScheduler(
            self._rng,
            config.warmup,
            config.fatigue,
            config.micro_pauses,
        )
        self._session_started = False
        self._last_click_time: float = 0.0

    @property
    def config(self) -> BehaviorConfig:
        return self._config

    @property
    def rng(self) -> SeededRNG:
        return self._rng

    def start_session(self) -> None:
        self._session.start_session()
        self._session_started = True
        self._last_click_time = time.perf_counter()

    def plan_action(
        self,
        current_x: float,
        current_y: float,
        target_x: float,
        target_y: float,
        target_width: float,
        target_metadata: Optional[dict] = None,
    ) -> BehavioralAction:
        if not self._session_started:
            self.start_session()

        session_enabled = self._config.warmup.enabled or self._config.fatigue.enabled
        if session_enabled:
            state = self._session.update()
        else:
            state = SessionState()

        jitter_x, jitter_y = 0.0, 0.0
        if self._config.jitter.enabled:
            jitter_x, jitter_y = apply_jitter(
                current_x, current_y,
                self._rng,
                self._config.jitter.amplitude_px,
                self._config.jitter.smoothness,
            )

        actual_target_x, actual_target_y = compute_endpoint_error(
            target_x, target_y,
            target_width,
            self._rng,
            self._config.endpoint_error.scale_px if self._config.endpoint_error.enabled else 0.0,
            self._config.endpoint_error.max_px if self._config.endpoint_error.enabled else 0.0,
        )

        distance = math.sqrt(
            (target_x - current_x) ** 2 + (target_y - current_y) ** 2
        )

        raw_reaction = sample_lognormal_bounded(
            self._rng,
            self._config.reaction_time.median_ms,
            self._config.reaction_time.sigma,
            self._config.reaction_time.minimum_ms,
            self._config.reaction_time.maximum_ms,
        )
        reaction_s = self._session.apply_modifier(raw_reaction, state) / 1000.0

        raw_movement = sample_fitts_mt(
            self._rng,
            distance,
            target_width,
            self._config.movement.a_ms,
            self._config.movement.b_ms,
            self._config.movement.variance,
        )
        movement_s = self._session.apply_modifier(raw_movement, state) / 1000.0

        raw_click = sample_lognormal_bounded(
            self._rng,
            self._config.click.median_ms,
            self._config.click.sigma,
            self._config.click.minimum_ms,
            self._config.click.maximum_ms,
        )
        click_s = raw_click / 1000.0

        trajectory = []
        if self._config.trajectory.enabled and movement_s > 0.001 and distance > 1.0:
            trajectory = generate_trajectory(
                jitter_x, jitter_y,
                actual_target_x, actual_target_y,
                movement_s,
                self._rng,
                num_points=self._config.trajectory.points,
                lateral_deviation_px=self._config.trajectory.lateral_deviation_px,
                velocity_variance=self._config.trajectory.velocity_variance,
                endpoint_correction=self._config.trajectory.endpoint_correction,
            )

        pause_s = 0.0
        micro_pause_ms = self._session.should_micro_pause()
        if micro_pause_ms is not None:
            pause_s = micro_pause_ms / 1000.0

        is_miss = False
        is_delayed = False
        if self._config.errors.enabled:
            is_miss = self._session.should_miss(
                self._config.errors.miss_probability, state
            )
            is_delayed = self._session.should_delay(
                self._config.errors.delayed_response_probability, state
            )
            if is_delayed:
                extra = self._rng.uniform(
                    self._config.errors.extra_pause_min_ms,
                    self._config.errors.extra_pause_max_ms,
                )
                pause_s += extra / 1000.0

        total_s = reaction_s + movement_s + click_s + pause_s

        endpoint_err = math.sqrt(
            (actual_target_x - target_x) ** 2 + (actual_target_y - target_y) ** 2
        )

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
            fitts_id=compute_fitts_id(distance, target_width),
            endpoint_error_px=endpoint_err,
            pause_duration_s=pause_s,
            is_miss=is_miss,
            is_delayed=is_delayed,
            session_modifier=state.modifier,
            warmup_factor=state.warmup_factor,
            fatigue_factor=state.fatigue_factor,
        )

    def wait_reaction(self, action: BehavioralAction) -> None:
        if action.reaction_time_s > 0:
            time.sleep(action.reaction_time_s)

    def execute_trajectory(
        self,
        action: BehavioralAction,
        move_fn,
        sleep_fn=None,
    ) -> None:
        if not action.trajectory:
            return

        if sleep_fn is None:
            sleep_fn = time.sleep

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
