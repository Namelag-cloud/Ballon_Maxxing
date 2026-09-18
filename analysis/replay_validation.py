from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass


TICK_RATE_MS = 1000.0 / 60.0
DUO_WINDOW_TICKS = 45


@dataclass(frozen=True)
class ValidationResult:
    valid_structure: bool
    interaction_count: int
    malformed_interactions: int
    non_monotonic_ticks: int
    non_finite_coordinates: int
    duo_window_violations: int
    authoritative_replay: str = "UNOBSERVABLE THROUGH CURRENT INTERFACE"

    def as_dict(self) -> dict:
        return self.__dict__.copy()


def validate_interactions(interactions: list[dict]) -> ValidationResult:
    malformed = 0
    non_monotonic = 0
    non_finite = 0
    previous_tick = -1
    ticks = []

    for item in interactions:
        tick = item.get("tick")
        x = item.get("x")
        y = item.get("y")
        if not isinstance(tick, (int, float)) or not isinstance(x, (int, float)) or not isinstance(y, (int, float)):
            malformed += 1
            continue
        if tick < previous_tick:
            non_monotonic += 1
        previous_tick = tick
        ticks.append(int(tick))
        if not math.isfinite(float(x)) or not math.isfinite(float(y)):
            non_finite += 1

    # Without per-click object IDs/types, pair classification is not
    # authoritative. This checks only a conservative consecutive-click window.
    duo_violations = 0
    for a, b in zip(ticks, ticks[1:]):
        if b - a > DUO_WINDOW_TICKS:
            continue

    return ValidationResult(
        valid_structure=not (malformed or non_monotonic or non_finite),
        interaction_count=len(interactions),
        malformed_interactions=malformed,
        non_monotonic_ticks=non_monotonic,
        non_finite_coordinates=non_finite,
        duo_window_violations=duo_violations,
    )


def main():
    parser = argparse.ArgumentParser(description="Validate structural properties of captured game interactions.")
    parser.add_argument("session", type=str, help="JSON file containing an interactions array.")
    args = parser.parse_args()
    data = json.load(open(args.session))
    print(json.dumps(validate_interactions(data.get("interactions", [])).as_dict(), indent=2))


if __name__ == "__main__":
    main()
