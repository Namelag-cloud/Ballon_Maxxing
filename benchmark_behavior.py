import sys
import os
import json
import math
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from behavior.config import (
    BehaviorConfig, WarmupConfig, FatigueConfig, MicroPauseConfig,
    EndpointErrorConfig,
)
from behavior.model import BehavioralModel
from behavior.distributions import SeededRNG, sample_lognormal_bounded, sample_fitts_mt
from behavior.movement import compute_fitts_id


def percentile(data, p):
    n = len(data)
    k = (n - 1) * p / 100.0
    f = int(k)
    c = min(f + 1, n - 1)
    return data[f] + (k - f) * (data[c] - data[f])


def compute_skewness(values, mean, std):
    if std == 0 or len(values) < 3:
        return 0.0
    n = len(values)
    m3 = sum((v - mean) ** 3 for v in values) / n
    return m3 / (std ** 3)


def compute_kurtosis_excess(values, mean, std):
    if std == 0 or len(values) < 4:
        return 0.0
    n = len(values)
    m4 = sum((v - mean) ** 4 for v in values) / n
    return m4 / (std ** 4) - 3.0


def compute_stats(values):
    n = len(values)
    mean = sum(values) / n
    variance = sum((v - mean) ** 2 for v in values) / n
    std = variance ** 0.5
    sorted_vals = sorted(values)

    return {
        "n": n,
        "mean": mean,
        "std": std,
        "min": sorted_vals[0],
        "max": sorted_vals[-1],
        "p1": percentile(sorted_vals, 1),
        "p5": percentile(sorted_vals, 5),
        "p25": percentile(sorted_vals, 25),
        "p50": percentile(sorted_vals, 50),
        "p75": percentile(sorted_vals, 75),
        "p95": percentile(sorted_vals, 95),
        "p99": percentile(sorted_vals, 99),
        "skewness": compute_skewness(sorted_vals, mean, std),
        "kurtosis_excess": compute_kurtosis_excess(sorted_vals, mean, std),
    }


def print_stats(name, stats_dict):
    print(f"\n{'='*55}")
    print(f"  {name}")
    print(f"{'='*55}")
    print(f"  n         = {stats_dict['n']:>10d}")
    print(f"  mean      = {stats_dict['mean']:>10.3f}")
    print(f"  std       = {stats_dict['std']:>10.3f}")
    print(f"  min       = {stats_dict['min']:>10.3f}")
    print(f"  max       = {stats_dict['max']:>10.3f}")
    print(f"  p1        = {stats_dict['p1']:>10.3f}")
    print(f"  p5        = {stats_dict['p5']:>10.3f}")
    print(f"  p25       = {stats_dict['p25']:>10.3f}")
    print(f"  p50       = {stats_dict['p50']:>10.3f}")
    print(f"  p75       = {stats_dict['p75']:>10.3f}")
    print(f"  p95       = {stats_dict['p95']:>10.3f}")
    print(f"  p99       = {stats_dict['p99']:>10.3f}")
    print(f"  skewness  = {stats_dict['skewness']:>10.4f}")
    print(f"  kurtosis  = {stats_dict['kurtosis_excess']:>10.4f}")


def compute_correlation(xs, ys):
    n = len(xs)
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    cov = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys)) / n
    std_x = (sum((x - mean_x) ** 2 for x in xs) / n) ** 0.5
    std_y = (sum((y - mean_y) ** 2 for y in ys) / n) ** 0.5
    if std_x == 0 or std_y == 0:
        return 0.0
    return cov / (std_x * std_y)


def verify_fitts_monotonicity():
    rng = SeededRNG(42)
    a_ms, b_ms = 80.0, 70.0
    variance = 0.15
    n_per_cell = 500

    distances = [50, 100, 200, 400, 800]
    widths = [20, 40, 80]

    results = []
    for dist in distances:
        for width in widths:
            samples = [sample_fitts_mt(rng, dist, width, a_ms, b_ms, variance) for _ in range(n_per_cell)]
            mean_mt = sum(samples) / len(samples)
            predicted = a_ms + b_ms * compute_fitts_id(dist, width)
            results.append({
                "distance": dist,
                "width": width,
                "fitts_id": compute_fitts_id(dist, width),
                "predicted_mt_ms": predicted,
                "sampled_mt_ms": mean_mt,
                "sampled_mt_std": (sum((s - mean_mt) ** 2 for s in samples) / len(samples)) ** 0.5,
            })

    monotonic_increasing_dist = True
    for width in widths:
        mts = [r["sampled_mt_ms"] for r in results if r["width"] == width]
        for i in range(1, len(mts)):
            if mts[i] < mts[i - 1]:
                monotonic_increasing_dist = False

    monotonic_decreasing_width = True
    for dist in distances:
        mts = [r["sampled_mt_ms"] for r in results if r["distance"] == dist]
        for i in range(1, len(mts)):
            if mts[i] > mts[i - 1]:
                monotonic_decreasing_width = False

    predicted_vals = [r["predicted_mt_ms"] for r in results]
    sampled_vals = [r["sampled_mt_ms"] for r in results]
    correlation = compute_correlation(predicted_vals, sampled_vals)

    return {
        "cells": results,
        "monotonic_increasing_distance": monotonic_increasing_dist,
        "monotonic_decreasing_width": monotonic_decreasing_width,
        "predicted_vs_sampled_correlation": correlation,
    }


def run_benchmark(n=100000, seed=12345):
    config = BehaviorConfig(
        seed=seed,
        warmup=WarmupConfig(enabled=False),
        fatigue=FatigueConfig(enabled=False),
        micro_pauses=MicroPauseConfig(enabled=False),
    )
    model = BehavioralModel(config)
    model.start_session()

    t0 = time.perf_counter()

    rt_values = []
    click_values = []
    mt_values = []
    total_values = []
    endpoint_errors = []
    endpoint_dx = []
    endpoint_dy = []

    target_x, target_y = 500.0, 500.0

    for i in range(n):
        action = model.plan_action(0, 0, target_x, target_y, 50)
        rt_values.append(action.reaction_time_s * 1000)
        click_values.append(action.click_duration_s * 1000)
        mt_values.append(action.movement_time_s * 1000)
        total_values.append(action.total_time_s * 1000)
        endpoint_errors.append(action.endpoint_error_px)
        click_x, click_y = action.click_position
        endpoint_dx.append(click_x - target_x)
        endpoint_dy.append(click_y - target_y)

    elapsed = time.perf_counter() - t0

    rt_stats = compute_stats(rt_values)
    click_stats = compute_stats(click_values)
    mt_stats = compute_stats(mt_values)
    total_stats = compute_stats(total_values)
    ep_stats = compute_stats(endpoint_errors)
    dx_stats = compute_stats(endpoint_dx)
    dy_stats = compute_stats(endpoint_dy)

    ep_exceed = {}
    for threshold in [1, 3, 5, 8, 10]:
        count = sum(1 for r in endpoint_errors if r > threshold)
        ep_exceed[f">{threshold}px"] = count / n * 100

    fitts = verify_fitts_monotonicity()

    result = {
        "config": {
            "seed": seed,
            "n_samples": n,
            "reaction_time": {
                "median_ms": config.reaction_time.median_ms,
                "sigma": config.reaction_time.sigma,
                "minimum_ms": config.reaction_time.minimum_ms,
                "maximum_ms": config.reaction_time.maximum_ms,
            },
            "movement": {
                "a_ms": config.movement.a_ms,
                "b_ms": config.movement.b_ms,
                "variance": config.movement.variance,
            },
            "click": {
                "median_ms": config.click.median_ms,
                "sigma": config.click.sigma,
                "minimum_ms": config.click.minimum_ms,
                "maximum_ms": config.click.maximum_ms,
            },
            "endpoint_error": {
                "enabled": config.endpoint_error.enabled,
                "scale_px": config.endpoint_error.scale_px,
                "max_px": config.endpoint_error.max_px,
            },
        },
        "reaction_time": rt_stats,
        "click_duration": click_stats,
        "movement_time": mt_stats,
        "total_action_time": total_stats,
        "endpoint_error": ep_stats,
        "endpoint_error_exceedance_pct": ep_exceed,
        "endpoint_dx": dx_stats,
        "endpoint_dy": dy_stats,
        "fitts_analysis": fitts,
        "benchmark_runtime_s": elapsed,
        "samples_per_second": n / elapsed,
    }

    return result


def print_summary(result):
    n = result["config"]["n_samples"]
    print(f"\n{'='*60}")
    print(f"  BEHAVIORAL MODEL BENCHMARK — {n:,} samples")
    print(f"{'='*60}")
    print(f"  Seed: {result['config']['seed']}")
    print(f"  Runtime: {result['benchmark_runtime_s']:.2f}s ({result['samples_per_second']:,.0f} samples/sec)")

    print_stats("Reaction Time (ms)", result["reaction_time"])
    print_stats("Click Duration (ms)", result["click_duration"])
    print_stats("Movement Time (ms)", result["movement_time"])
    print_stats("Total Action Time (ms)", result["total_action_time"])
    print_stats("Endpoint Error Radial (px)", result["endpoint_error"])
    print_stats("Endpoint Error DX (px)", result["endpoint_dx"])
    print_stats("Endpoint Error DY (px)", result["endpoint_dy"])

    print(f"\n{'='*55}")
    print(f"  Endpoint Error Exceedance (%)")
    print(f"{'='*55}")
    for label, pct in result["endpoint_error_exceedance_pct"].items():
        print(f"  {label:>8s}  {pct:6.2f}%")

    fitts = result["fitts_analysis"]
    print(f"\n{'='*55}")
    print(f"  Fitts Law Analysis")
    print(f"{'='*55}")
    print(f"  {'Dist':>6} {'Width':>6} {'ID':>7} {'Pred_MT':>9} {'Samp_MT':>9} {'Samp_Std':>9}")
    print(f"  {'-'*50}")
    for cell in fitts["cells"]:
        print(f"  {cell['distance']:6d} {cell['width']:6d} {cell['fitts_id']:7.2f} "
              f"{cell['predicted_mt_ms']:9.2f} {cell['sampled_mt_ms']:9.2f} {cell['sampled_mt_std']:9.2f}")

    print(f"\n  Monotonic dist up -> MT up : {'PASS' if fitts['monotonic_increasing_distance'] else 'FAIL'}")
    print(f"  Monotonic width down -> MT up : {'PASS' if fitts['monotonic_decreasing_width'] else 'FAIL'}")
    print(f"  Predicted vs Sampled r : {fitts['predicted_vs_sampled_correlation']:.4f}")
    print()


def main():
    n = 100000
    seed = 12345

    result = run_benchmark(n=n, seed=seed)
    print_summary(result)

    output_dir = Path("telemetry_output")
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / f"behavior_benchmark_{seed}.json"
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Results saved to: {output_path}")


if __name__ == "__main__":
    main()
