from __future__ import annotations

import argparse
import json
from pathlib import Path

LABELS = (
    "balloon",
    "duo",
    "super_g_ring",
    "notification",
    "ui",
    "unknown",
)
STATS_KEYS = LABELS + ("skipped", "total")


def discover_label_files(root: Path) -> list[Path]:
    """Find per-session human-label files under a raw dataset root."""
    if not root.is_dir():
        raise FileNotFoundError(f"Training root not found: {root}")
    return sorted(root.glob("*/labels.json"))


def dataset_counts(root: Path) -> dict[str, int]:
    """Count human labels across all session labels.json files."""
    counts = {key: 0 for key in STATS_KEYS}

    for path in discover_label_files(root):
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError(f"Expected a JSON object in {path}")

        for value in data.values():
            if value in LABELS:
                counts[value] += 1
            elif value == "skip":
                counts["skipped"] += 1
            else:
                raise ValueError(f"Unknown label {value!r} in {path}")

    counts["total"] = sum(counts[key] for key in STATS_KEYS[:-1])
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Print human-labeled candidate dataset class counts."
    )
    parser.add_argument(
        "path",
        nargs="?",
        type=Path,
        default=Path("training/raw"),
        help="raw dataset root containing per-session labels.json files",
    )
    args = parser.parse_args()

    counts = dataset_counts(args.path)
    for key in STATS_KEYS:
        print(f"{key}: {counts[key]}")


if __name__ == "__main__":
    main()
