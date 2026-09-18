from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Print the prescribed live calibration plan.")
    parser.add_argument("--dry-runs", type=int, default=200)
    parser.add_argument("--live-runs", type=int, default=50)
    parser.add_argument("--duration", type=float, default=60.0)
    parser.add_argument("--output", type=Path, default=Path("calibration_runs"))
    args = parser.parse_args()

    print("Perception-only calibration")
    print(f"  sessions: {args.dry_runs}")
    print(f"  command: python main.py --mode dry-run --game balloons --duration {args.duration} --no-gui")
    print(f"  output directory: {args.output}")
    print()
    print("Bot behavioral calibration")
    print(f"  sessions: {args.live_runs}")
    print(f"  command: python main.py --mode live --game balloons --duration {args.duration}")
    print("  Keep Chrome visible and use --startup-delay to switch to the game.")
    print()
    print("Human reference dataset")
    print("  Collect the same outcome fields from explicitly labeled human sessions.")
    print("  Do not treat bot-generated telemetry as human ground truth.")
    print()
    print("Analysis")
    print("  Report reaction median/IQR, normalized spatial error, Duo delay median/IQR,")
    print("  observable hit/miss and notification rates, session duration, click-tick modulo 3,")
    print("  lag-1 reaction autocorrelation, and intended-vs-observed timing error.")
    print("  Missing signals must remain UNOBSERVABLE THROUGH CURRENT INTERFACE.")
    print("  python -m analysis.behavior_validation calibration_runs/*.json --reference human_reference/*.json")
    print()
    print("Pixel feedback")
    print("  Configure feedback.combo_roi, feedback.score_roi and feedback.pop_roi")
    print("  only after visually validating the corresponding UI regions.")
    print("  Unconfigured signals remain UNOBSERVABLE THROUGH CURRENT INTERFACE.")


if __name__ == "__main__":
    main()
