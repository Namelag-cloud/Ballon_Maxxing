from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean, median, pstdev


def numeric_stats(values):
    values = [float(v) for v in values if v is not None]
    if not values:
        return {"count": 0}
    ordered = sorted(values)
    n = len(ordered)
    return {
        "count": n,
        "mean": mean(values),
        "median": median(values),
        "stdev": pstdev(values) if n > 1 else 0.0,
        "min": min(values),
        "max": max(values),
        "p95": ordered[max(0, math.ceil(n * .95) - 1)],
    }


def iqr(values):
    values = sorted(float(v) for v in values)
    if len(values) < 2:
        return None
    q1 = values[int(.25 * (len(values) - 1))]
    q3 = values[int(.75 * (len(values) - 1))]
    return q3 - q1


def load_rows(paths):
    rows = []
    for path in paths:
        path = Path(path)
        if path.suffix.lower() == ".json":
            data = json.loads(path.read_text())
            rows.extend(data.get("outcomes", []))
        else:
            with open(path, newline="") as f:
                rows.extend(csv.DictReader(f))
    return rows


def values(rows, key):
    result = []
    for row in rows:
        value = row.get(key)
        if value not in (None, ""):
            try:
                result.append(float(value))
            except (ValueError, TypeError):
                pass
    return result


def summarize(rows):
    mapping = {
        "reaction_interval_ticks": "estimated_reaction_ticks",
        "spatial_error_radius": "endpoint_error_px",
        "duo_delta_ms": "duo_delta_ms",
        "click_tick_modulo": "click_tick_modulo",
        "timing_error_ms": "timing_error_ms",
    }
    result = {}
    for metric, key in mapping.items():
        vals = values(rows, key)
        stats = numeric_stats(vals)
        stats["iqr"] = iqr(vals)
        result[metric] = stats

    for metric, key in (("hit_rate", "hit"), ("miss_rate", "miss"), ("notification_rate", "notification")):
        vals = values(rows, key)
        result[metric] = {"count": len(vals), "mean": mean(vals) if vals else None}

    return result


def ks_distance(a, b):
    a, b = sorted(a), sorted(b)
    if not a or not b:
        return None
    points = sorted(set(a + b))
    ia = ib = 0
    best = 0.0
    for x in points:
        while ia < len(a) and a[ia] <= x:
            ia += 1
        while ib < len(b) and b[ib] <= x:
            ib += 1
        best = max(best, abs(ia / len(a) - ib / len(b)))
    return best


def compare(bot_rows, reference_rows):
    keys = {
        "reaction_interval_ticks": "estimated_reaction_ticks",
        "spatial_error_radius": "endpoint_error_px",
        "duo_delta_ms": "duo_delta_ms",
    }
    result = {}
    for metric, key in keys.items():
        bot = values(bot_rows, key)
        ref = values(reference_rows, key)
        result[metric] = {
            "bot": numeric_stats(bot),
            "reference": numeric_stats(ref),
            "bot_iqr": iqr(bot),
            "reference_iqr": iqr(ref),
            "ks_distance": ks_distance(bot, ref),
            "note": "KS distance is descriptive; it is not an acceptance criterion.",
        }
    return result


def main():
    parser = argparse.ArgumentParser(description="Summarize and compare behavioral telemetry.")
    parser.add_argument("bot", nargs="+", type=Path)
    parser.add_argument("--reference", nargs="+", type=Path)
    parser.add_argument("--json", dest="json_path", type=Path)
    args = parser.parse_args()

    bot_rows = load_rows(args.bot)
    output = {"bot": summarize(bot_rows)}
    if args.reference:
        output["comparison"] = compare(bot_rows, load_rows(args.reference))

    rendered = json.dumps(output, indent=2)
    if args.json_path:
        args.json_path.write_text(rendered)
    else:
        print(rendered)


if __name__ == "__main__":
    main()
