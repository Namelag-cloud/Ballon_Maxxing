from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from training.dataset_stats import LABELS, discover_label_files

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15
SPLITS = ("train", "val", "test")


@dataclass(frozen=True)
class Sample:
    session: str
    image: str
    label: str


def load_samples(root: Path) -> list[Sample]:
    samples: list[Sample] = []
    for labels_path in discover_label_files(root):
        session_dir = labels_path.parent
        data = json.loads(labels_path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError(f"Expected a JSON object in {labels_path}")
        for image, label in data.items():
            if label not in LABELS:
                if label == "skip":
                    continue
                raise ValueError(f"Unknown label {label!r} in {labels_path}")
            image_path = session_dir / "crops" / image
            if not image_path.is_file():
                # Older datasets may store the crop path directly in the key.
                image_path = session_dir / image
            if not image_path.is_file():
                raise FileNotFoundError(f"Labeled crop not found: {image_path}")
            samples.append(
                Sample(
                    session=session_dir.name,
                    image=str(image_path),
                    label=label,
                )
            )
    return samples


def session_audit(samples: Iterable[Sample]) -> dict:
    by_session: dict[str, Counter[str]] = defaultdict(Counter)
    for sample in samples:
        by_session[sample.session][sample.label] += 1
    return {
        "sessions": len(by_session),
        "samples": sum(sum(c.values()) for c in by_session.values()),
        "class_counts": dict(Counter(s.label for s in samples)),
        "session_counts": {
            session: dict(sorted(counts.items()))
            for session, counts in sorted(by_session.items())
        },
    }


def _score_assignment(
    assignment: dict[str, str],
    session_counts: dict[str, Counter[str]],
    targets: dict[str, float],
    total_by_class: Counter[str],
) -> float:
    score = 0.0
    for split in SPLITS:
        split_counts = Counter()
        for session, assigned in assignment.items():
            if assigned == split:
                split_counts.update(session_counts[session])
        split_total = sum(split_counts.values())
        target_total = targets[split]
        if split_total:
            score += abs(split_total / sum(total_by_class.values()) - target_total) * 3.0
        for label, total in total_by_class.items():
            desired = total * target_total
            score += abs(split_counts[label] - desired) / max(1.0, total)
            # Prefer at least one evaluation example when the data permits it.
            if split in ("val", "test") and total >= 3 and split_counts[label] == 0:
                score += 2.0
    return score


def split_sessions(
    samples: list[Sample],
    *,
    seed: int = 1337,
    train_ratio: float = TRAIN_RATIO,
    val_ratio: float = VAL_RATIO,
) -> dict[str, list[Sample]]:
    if not samples:
        raise ValueError("No labeled samples found")
    test_ratio = 1.0 - train_ratio - val_ratio
    if min(train_ratio, val_ratio, test_ratio) <= 0:
        raise ValueError("Split ratios must be positive")

    session_counts: dict[str, Counter[str]] = defaultdict(Counter)
    for sample in samples:
        session_counts[sample.session][sample.label] += 1
    sessions = sorted(session_counts)
    if len(sessions) < 3:
        raise ValueError(
            f"Need at least 3 sessions for train/val/test; found {len(sessions)}"
        )

    ratios = {"train": train_ratio, "val": val_ratio, "test": test_ratio}
    total_by_class = Counter(s.label for s in samples)
    rng = random.Random(seed)

    best_assignment: dict[str, str] | None = None
    best_score = float("inf")
    # Randomized greedy search is deliberately deterministic from the seed.
    # It gives better class balance than simply sorting sessions by size.
    for _ in range(max(1000, len(sessions) * 200)):
        order = sessions[:]
        rng.shuffle(order)
        assignment: dict[str, str] = {}
        running = Counter()
        split_sizes = Counter()
        for session in order:
            candidates = list(SPLITS)
            rng.shuffle(candidates)
            chosen = min(
                candidates,
                key=lambda split: (
                    split_sizes[split] / max(1, len(samples) * ratios[split]),
                    sum(
                        abs(
                            running[(split, label)]
                            + session_counts[session][label]
                            - total_by_class[label] * ratios[split]
                        )
                        / max(1, total_by_class[label])
                        for label in total_by_class
                    ),
                ),
            )
            assignment[session] = chosen
            split_sizes[chosen] += sum(session_counts[session].values())
            for label, count in session_counts[session].items():
                running[(chosen, label)] += count

        # Reject empty splits.
        if set(assignment.values()) != set(SPLITS):
            continue
        score = _score_assignment(
            assignment, session_counts, ratios, total_by_class
        )
        if score < best_score:
            best_score = score
            best_assignment = assignment

    if best_assignment is None:
        raise RuntimeError("Could not construct a non-empty session split")

    result = {split: [] for split in SPLITS}
    for sample in samples:
        result[best_assignment[sample.session]].append(sample)
    return result


def write_manifests(
    splits: dict[str, list[Sample]],
    output_dir: Path,
    *,
    seed: int,
) -> dict[str, dict]:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {"seed": seed, "splits": {}}
    for split, samples in splits.items():
        path = output_dir / f"{split}.jsonl"
        with path.open("w", encoding="utf-8") as handle:
            for sample in samples:
                handle.write(
                    json.dumps(
                        {
                            "session": sample.session,
                            "image": sample.image,
                            "label": sample.label,
                        },
                        separators=(",", ":"),
                    )
                    + "\n"
                )
        summary["splits"][split] = {
            "samples": len(samples),
            "sessions": sorted({s.session for s in samples}),
            "class_counts": dict(sorted(Counter(s.label for s in samples).items())),
            "manifest": str(path),
        }
    (output_dir / "split_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a session-aware ML split.")
    parser.add_argument("root", nargs="?", type=Path, default=Path("training/raw"))
    parser.add_argument("--output", type=Path, default=Path("training/artifacts/splits"))
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()

    samples = load_samples(args.root)
    audit = session_audit(samples)
    splits = split_sessions(samples, seed=args.seed)
    summary = write_manifests(splits, args.output, seed=args.seed)
    print(json.dumps({"audit": audit, "summary": summary}, indent=2))


if __name__ == "__main__":
    main()
