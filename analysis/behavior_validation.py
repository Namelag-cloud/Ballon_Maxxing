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
            duration = data.get("session", {}).get("duration_s")
            outcomes = data.get("outcomes", [])
            for index, outcome in enumerate(outcomes):
                row = dict(outcome)
                if duration is not None and index == 0:
                    row["_session_duration_s"] = duration
                rows.append(row)
            if duration is not None and not outcomes:
                rows.append({"_session_duration_s": duration})
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


def binary_values(rows, key):
    result = []
    for row in rows:
        value = row.get(key)
        if isinstance(value, bool):
            result.append(float(value))
        elif isinstance(value, (int, float)):
            result.append(float(value))
        elif isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {"true", "1"}:
                result.append(1.0)
            elif normalized in {"false", "0"}:
                result.append(0.0)
    return result


def lag1_autocorrelation(values):
    if len(values) < 3:
        return None
    x = [float(v) for v in values[:-1]]
    y = [float(v) for v in values[1:]]
    mx, my = mean(x), mean(y)
    num = sum((a - mx) * (b - my) for a, b in zip(x, y))
    denx = sum((a - mx) ** 2 for a in x)
    deny = sum((b - my) ** 2 for b in y)
    return num / math.sqrt(denx * deny) if denx and deny else None


def summarize(rows):
    reaction = values(rows, "estimated_acquisition_click_ticks")
    spatial = values(rows, "endpoint_error_px")
    normalized_spatial = []
    for row in rows:
        try:
            error = float(row.get("endpoint_error_px"))
            radius = float(row.get("target_radius_px"))
            if radius > 0:
                normalized_spatial.append(error / radius)
        except (TypeError, ValueError):
            continue
    duo = values(rows, "duo_delta_ms")
    timing_error = values(rows, "intended_vs_observed_timing_error_ms")
    durations = values(rows, "_session_duration_s")
    modulo = values(rows, "click_tick_modulo")

    modulo_distribution = {str(i): modulo.count(float(i)) for i in range(3)}
    result = {
        "reaction_interval_ticks": {**numeric_stats(reaction), "iqr": iqr(reaction)},
        "spatial_error_px": {**numeric_stats(spatial), "iqr": iqr(spatial)},
        "spatial_error_normalized_by_target_radius": {
            **numeric_stats(normalized_spatial), "iqr": iqr(normalized_spatial)
        },
        "duo_delay_ms": {**numeric_stats(duo), "iqr": iqr(duo)},
        "session_duration_s": {**numeric_stats(durations), "iqr": iqr(durations)},
        "click_tick_modulo_3": {**numeric_stats(modulo), "iqr": iqr(modulo), "distribution": modulo_distribution},
        "lag1_reaction_autocorrelation": lag1_autocorrelation(reaction),
        "intended_vs_observed_timing_error_ms": {
            **numeric_stats(timing_error), "iqr": iqr(timing_error)
        },
    }

    for metric, key in (("hit_rate", "hit"), ("miss_rate", "miss"), ("notification_rate", "notification")):
        vals = binary_values(rows, key)
        result[metric] = {"count": len(vals), "mean": mean(vals) if vals else None}

    if not reaction:
        result["reaction_interval_ticks"]["observability"] = "UNOBSERVABLE THROUGH CURRENT INTERFACE"
    if not normalized_spatial:
        result["spatial_error_normalized_by_target_radius"]["observability"] = "UNOBSERVABLE THROUGH CURRENT INTERFACE"
    if not duo:
        result["duo_delay_ms"]["observability"] = "UNOBSERVABLE THROUGH CURRENT INTERFACE"
    if not binary_values(rows, "hit"):
        result["hit_rate"]["observability"] = "UNOBSERVABLE THROUGH CURRENT INTERFACE"
        result["miss_rate"]["observability"] = "UNOBSERVABLE THROUGH CURRENT INTERFACE"
    if not binary_values(rows, "notification"):
        result["notification_rate"]["observability"] = "UNOBSERVABLE THROUGH CURRENT INTERFACE"
    if not durations:
        result["session_duration_s"]["observability"] = "UNOBSERVABLE THROUGH CURRENT INTERFACE"
    if not modulo:
        result["click_tick_modulo_3"]["observability"] = "UNOBSERVABLE THROUGH CURRENT INTERFACE"
    if not timing_error:
        result["intended_vs_observed_timing_error_ms"]["observability"] = "UNOBSERVABLE THROUGH CURRENT INTERFACE"

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


def metric_values(rows, key):
    if key != "spatial_error_normalized_by_target_radius":
        return values(rows, key)
    result = []
    for row in rows:
        try:
            radius = float(row.get("target_radius_px"))
            if radius > 0:
                result.append(float(row.get("endpoint_error_px")) / radius)
        except (TypeError, ValueError):
            continue
    return result


def compare(bot_rows, reference_rows):
    keys = {
        "reaction_interval_ticks": "estimated_acquisition_click_ticks",
        "spatial_error_radius": "endpoint_error_px",
        "duo_delta_ms": "duo_delta_ms",
    }
    result = {}
    for metric, key in keys.items():
        bot = metric_values(bot_rows, key)
        ref = metric_values(reference_rows, key)
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
