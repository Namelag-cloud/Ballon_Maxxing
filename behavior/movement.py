from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

from behavior.distributions import SeededRNG


@dataclass
class TrajectoryPoint:
    x: float
    y: float
    t: float
    velocity: float


@dataclass
class MovementPlan:
    points: list[TrajectoryPoint]
    total_time_s: float
    distance: float
    target_width: float
    fitts_id: float


def compute_fitts_id(distance: float, width: float) -> float:
    if distance <= 0:
        return 0.0
    w = max(width, 1.0)
    return math.log2(1.0 + distance / w)


def generate_trajectory(
    start_x: float,
    start_y: float,
    end_x: float,
    end_y: float,
    total_time_s: float,
    rng: SeededRNG,
    num_points: int = 20,
    lateral_deviation_px: float = 2.0,
    velocity_variance: float = 0.15,
    endpoint_correction: bool = True,
) -> list[TrajectoryPoint]:
    if total_time_s <= 0:
        return [TrajectoryPoint(end_x, end_y, 0.0, 0.0)]

    dx = end_x - start_x
    dy = end_y - start_y
    distance = math.sqrt(dx * dx + dy * dy)

    if distance < 0.5:
        return [TrajectoryPoint(end_x, end_y, 0.0, 0.0)]

    direction_x = dx / distance
    direction_y = dy / distance
    perp_x = -direction_y
    perp_y = direction_x

    n_segments = num_points - 1

    if velocity_variance > 0:
        raw_weights = []
        for i in range(n_segments):
            w = max(0.1, 1.0 + rng.normal(0, velocity_variance * 0.3))
            raw_weights.append(w)
        total_weight = sum(raw_weights)
        segment_times = [(w / total_weight) * total_time_s for w in raw_weights]
    else:
        segment_times = [total_time_s / n_segments] * n_segments

    cumulative_t = 0.0
    points = []

    for i in range(num_points):
        if i == 0:
            frac = 0.0
        elif i == num_points - 1:
            frac = 1.0
        else:
            frac = i / (num_points - 1)

        progress = _ease_in_out(frac)

        base_x = start_x + dx * progress
        base_y = start_y + dy * progress

        noise_scale = math.sin(math.pi * frac) * lateral_deviation_px
        lateral_offset = rng.normal(0, noise_scale) if noise_scale > 0.1 else 0.0

        x = base_x + perp_x * lateral_offset
        y = base_y + perp_y * lateral_offset

        if i > 0:
            cumulative_t += segment_times[i - 1]

        points.append(TrajectoryPoint(x=x, y=y, t=cumulative_t, velocity=0.0))

    for i in range(1, len(points) - 1):
        dt = points[i + 1].t - points[i - 1].t
        if dt > 0:
            dist = math.sqrt(
                (points[i + 1].x - points[i - 1].x) ** 2
                + (points[i + 1].y - points[i - 1].y) ** 2
            )
            points[i] = TrajectoryPoint(
                x=points[i].x, y=points[i].y, t=points[i].t, velocity=dist / dt
            )

    if endpoint_correction and len(points) >= 2:
        last = points[-1]
        second_last = points[-2]
        correction_strength = 0.7
        corrected_x = last.x + (end_x - last.x) * correction_strength
        corrected_y = last.y + (end_y - last.y) * correction_strength
        points[-1] = TrajectoryPoint(
            x=corrected_x,
            y=corrected_y,
            t=total_time_s,
            velocity=0.0,
        )
        dist = math.sqrt(
            (points[-1].x - second_last.x) ** 2
            + (points[-1].y - second_last.y) ** 2
        )
        dt = points[-1].t - second_last.t
        if dt > 0:
            points[-2] = TrajectoryPoint(
                x=second_last.x, y=second_last.y, t=second_last.t, velocity=dist / dt
            )

    return points


def _ease_in_out(t: float) -> float:
    if t < 0.5:
        return 2.0 * t * t
    else:
        return 1.0 - (-2.0 * t + 2.0) ** 2 / 2.0


def apply_jitter(
    x: float,
    y: float,
    rng: SeededRNG,
    amplitude_px: float = 1.5,
    smoothness: float = 0.8,
) -> tuple[float, float]:
    if amplitude_px <= 0:
        return x, y

    jx = rng.normal(0, amplitude_px)
    jy = rng.normal(0, amplitude_px)

    if smoothness > 0:
        scale = amplitude_px * (1.0 - smoothness * 0.5)
        jx *= scale / max(abs(jx), 0.1)
        jy *= scale / max(abs(jy), 0.1)

    return x + jx, y + jy


def compute_endpoint_error(
    target_x: float,
    target_y: float,
    target_width: float,
    rng: SeededRNG,
    scale_px: float = 0.0,
    max_px: float = 0.0,
) -> tuple[float, float]:
    if scale_px <= 0:
        return target_x, target_y

    r = rng.rayleigh(scale_px)

    if max_px > 0:
        r = min(r, max_px)

    theta = rng.uniform(0, 2.0 * math.pi)
    dx = r * math.cos(theta)
    dy = r * math.sin(theta)

    return target_x + dx, target_y + dy
