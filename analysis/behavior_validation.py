from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean, median, pstdev


METRICS = (
    "reaction_ticks",
    "spatial_error_radius",
    "hit_rate",
    "miss_rate",
    "notification_rate",
    "duo_delta_ms",
    "super_g_outcome_rate",
    "balloon_g_abandonment",
    "session_duration_s",
    "idle_gap_s",
    "click_tick_modulo",
    "reaction_lag1_autocorrelation",
    "intended_observed_timing_error_ms",
)


def numeric_stats(values):
    values = [float(v) for v in values if v is not None]
    if not values:
        return {"count": 0}
    return {
        "count": len(values),
        "mean": mean(values),
        "median": median(values),
        "stdev": pstdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
        "p95": sorted(values)[max(0, math.ceil(len(values) * .95) - 1)],
    }


def load_rows(paths):
    rows = []
    for path in paths:
        with open(path, newline="") as f:
            rows.extend(csv.DictReader(f))
    return rows


def summarize(rows):
    mapping = {
        "reaction_ticks": "observed_reaction_ticks",
        "spatial_error_radius": "spatial_error_radius",
        "hit_rate": "hit",
        "miss_rate": "miss",
        "notification_rate": "notification",
        "duo_delta_ms": "duo_delta_ms",
        "session_duration_s": "session_duration_s",
        "idle_gap_s": "idle_gap_s",
        "click_tick_modulo": "click_tick_modulo",
        "intended_observed_timing_error_ms": "timing_error_ms",
    }
    result = {}
    for metric, key in mapping.items():
        vals = []
        for row in rows:
            value = row.get(key)
            if value not in (None, ""):
                try:
                    vals.append(float(value))
                except ValueError:
                    pass
        result[metric] = numeric_stats(vals)
    return result


def main():
    parser = argparse.ArgumentParser(description="Summarize bot/human behavioral telemetry.")
    parser.add_argument("csv", nargs="+", type=Path)
    parser.add_argument("--json", dest="json_path", type=Path)
    args = parser.parse_args()
    summary = summarize(load_rows(args.csv))
    output = json.dumps(summary, indent=2)
    if args.json_path:
        args.json_path.write_text(output)
    else:
        print(output)


if __name__ == "__main__":
    main()
