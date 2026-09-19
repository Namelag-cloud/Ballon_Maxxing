from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2


LABEL_KEYS = {
    ord("b"): "balloon",
    ord("u"): "ui",
    ord("n"): "notification",
    ord("d"): "duo",
    ord("g"): "super_g_ring",
    ord("x"): "unknown",
    ord("s"): "skip",
}
LABEL_NAMES = tuple(dict.fromkeys(LABEL_KEYS.values()))
_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}


def _atomic_write_json(path: Path, payload: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


class LabelStore:
    """Persist human labels so annotation can be interrupted and resumed."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.labels: dict[str, str] = {}
        if self.path.exists():
            data = json.loads(self.path.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                raise ValueError(f"Expected a JSON object in {self.path}")
            self.labels = {str(k): str(v) for k, v in data.items()}
            invalid = set(self.labels.values()) - set(LABEL_NAMES)
            if invalid:
                raise ValueError(f"Unknown labels in {self.path}: {sorted(invalid)}")

    def set(self, crop: str, label: str) -> None:
        if label not in LABEL_NAMES:
            raise ValueError(f"Unknown label: {label}")
        self.labels[crop] = label
        _atomic_write_json(self.path, self.labels)

    def get(self, crop: str) -> str | None:
        return self.labels.get(crop)

    def counts(self) -> dict[str, int]:
        return {label: sum(value == label for value in self.labels.values()) for label in LABEL_NAMES}


def load_crops(session_dir: Path) -> list[Path]:
    crop_dir = session_dir / "crops"
    if not crop_dir.is_dir():
        raise FileNotFoundError(f"Crop directory not found: {crop_dir}")
    return sorted(p for p in crop_dir.iterdir() if p.is_file() and p.suffix.lower() in _IMAGE_EXTENSIONS)


def discover_sessions(root: Path) -> list[Path]:
    """Return every session directory containing a crops directory."""
    if not root.is_dir():
        raise FileNotFoundError(f"Training root not found: {root}")
    return sorted(
        path
        for path in root.iterdir()
        if path.is_dir() and (path / "crops").is_dir()
    )


def _draw_overlay(
    image,
    index: int,
    total: int,
    crop: Path,
    label: str | None,
    counts: dict[str, int],
    session_dir: Path,
) -> Any:
    display = image.copy()
    height, width = display.shape[:2]
    scale = max(1.0, min(5.0, 700.0 / max(height, width)))
    if scale != 1.0:
        display = cv2.resize(display, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
    canvas = cv2.copyMakeBorder(display, 0, 114, 0, 0, cv2.BORDER_CONSTANT, value=(20, 20, 20))
    lines = [
        f"{index + 1}/{total}  {session_dir.name}  {crop.name}",
        f"Current: {label or 'unlabeled'}",
        "B balloon  D duo  G Super-G ring  U ui  N notification  X unknown  S skip",
        "Left/Right navigate  Q quit/save",
        " | ".join(f"{name}:{counts[name]}" for name in LABEL_NAMES),
    ]
    for row, line in enumerate(lines):
        cv2.putText(
            canvas,
            line,
            (10, display.shape[0] + 20 + row * 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (235, 235, 235),
            1,
            cv2.LINE_AA,
        )
    return canvas


def _dataset_counts(entries: list[tuple[Path, Path, LabelStore]]) -> dict[str, int]:
    """Count each label once per labeled crop, not once per entry referencing its store."""
    counts = {label: 0 for label in LABEL_NAMES}
    seen: set[tuple[Path, str]] = set()
    for crop, session_dir, store in entries:
        key = (session_dir, crop.name)
        if key in seen:
            continue
        seen.add(key)
        label = store.get(crop.name)
        if label is not None:
            counts[label] += 1
    return counts


def _label_entries(entries: list[tuple[Path, Path, LabelStore]], *, window_name: str) -> int:
    if not entries:
        return 0

    index = next(
        (i for i, (crop, _, store) in enumerate(entries) if store.get(crop.name) is None),
        0,
    )

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 1)

    try:
        while True:
            crop, session_dir, store = entries[index]
            image = cv2.imread(str(crop))
            if image is None:
                raise RuntimeError(f"Could not read crop: {crop}")

            counts = _dataset_counts(entries)
            canvas = _draw_overlay(
                image,
                index,
                len(entries),
                crop,
                store.get(crop.name),
                counts,
                session_dir,
            )
            cv2.imshow(window_name, canvas)
            key = cv2.waitKeyEx(0)

            if key in (ord("q"), ord("Q"), 27):
                return sum(_dataset_counts(entries).values())

            if key in LABEL_KEYS or key in {ord(c.upper()) for c in "bdgnuxs"}:
                label = LABEL_KEYS.get(key)
                if label is None:
                    label = LABEL_KEYS.get(key + 32)
                if label is not None:
                    store.set(crop.name, label)
                    if index < len(entries) - 1:
                        index += 1
                    continue

            if key in (81, 2424832):  # left arrow on common OpenCV backends
                index = max(0, index - 1)
                continue

            if key in (83, 2555904):  # right arrow on common OpenCV backends
                index = min(len(entries) - 1, index + 1)
                continue
    finally:
        cv2.destroyWindow(window_name)


def label_session(session_dir: Path, *, window_name: str = "Candidate Labeler") -> int:
    """Label one session. Kept for backwards-compatible single-session use."""
    crops = load_crops(session_dir)
    if not crops:
        raise ValueError(f"No candidate crops found in {session_dir / 'crops'}")
    store = LabelStore(session_dir / "labels.json")
    entries = [(crop, session_dir, store) for crop in crops]
    return _label_entries(entries, window_name=window_name)


def label_root(root: Path, *, window_name: str = "Candidate Labeler") -> int:
    """Label all candidate crops across every session under root."""
    sessions = discover_sessions(root)
    if not sessions:
        raise ValueError(f"No session directories with crops found in {root}")

    entries: list[tuple[Path, Path, LabelStore]] = []
    for session_dir in sessions:
        store = LabelStore(session_dir / "labels.json")
        crops = load_crops(session_dir)
        entries.extend((crop, session_dir, store) for crop in crops)

    if not entries:
        raise ValueError(f"No candidate crops found under {root}")

    return _label_entries(entries, window_name=window_name)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Human-label candidate crops across one session or the whole training/raw dataset."
    )
    parser.add_argument(
        "path",
        nargs="?",
        type=Path,
        default=Path("training/raw"),
        help="session directory or dataset root (default: training/raw)",
    )
    args = parser.parse_args()

    if (args.path / "crops").is_dir():
        labeled = label_session(args.path)
        print(f"Labeling saved: {args.path / 'labels.json'}")
    else:
        labeled = label_root(args.path)
        print(f"Dataset labeling saved under: {args.path}")

    print(f"Labeled/skipped entries: {labeled}")


if __name__ == "__main__":
    main()
