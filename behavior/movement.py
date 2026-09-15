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

    points = []
    cumulative_t = 0.0

    for i in range(num_points):
        if i == 0:
            frac = 0.0
        elif i == num_points - 1:
            frac = 1.0
        else:
            frac = i / (num_points - 1)

        base_x = start_x + dx * frac
        base_y = start_y + dy * frac

        progress_curve = _ease_in_out(frac)
        noise_scale = math.sin(math.pi * frac) * lateral_deviation_px
        lateral_offset = rng.normal(0, noise_scale) if noise_scale > 0.1 else 0.0

        x = base_x + perp_x * lateral_offset
        y = base_y + perp_y * lateral_offset

        segment_time = total_time_s / (num_points - 1)
        if velocity_variance > 0:
            speed_factor = 1.0 + rng.normal(0, velocity_variance * 0.3)
            speed_factor = max(0.5, min(2.0, speed_factor))
            segment_time *= speed_factor

        cumulative_t += segment_time

        if i == 0:
            velocity = 0.0
        elif i == num_points - 1:
            velocity = 0.0
        else:
            velocity = distance / total_time_s if total_time_s > 0 else 0.0

        points.append(TrajectoryPoint(x=x, y=y, t=cumulative_t, velocity=velocity))

    if endpoint_correction and points:
        last = points[-1]
        correction_strength = 0.7
        points[-1] = TrajectoryPoint(
            x=last.x + (end_x - last.x) * correction_strength,
            y=last.y + (end_y - last.y) * correction_strength,
            t=last.t,
            velocity=0.0,
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
