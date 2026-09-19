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
    return sorted(p for p in crop_dir.iterdir() if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg"})


def _draw_overlay(image, index: int, total: int, crop: Path, label: str | None, counts: dict[str, int]) -> Any:
    display = image.copy()
    height, width = display.shape[:2]
    scale = max(1.0, min(5.0, 700.0 / max(height, width)))
    if scale != 1.0:
        display = cv2.resize(display, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
    canvas = cv2.copyMakeBorder(display, 0, 96, 0, 0, cv2.BORDER_CONSTANT, value=(20, 20, 20))
    lines = [
        f"{index + 1}/{total}  {crop.name}",
        f"Current: {label or 'unlabeled'}",
        f"B balloon  U ui  N notification  X unknown  S skip",
        f"Left/Right navigate  Q quit/save",
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


def label_session(session_dir: Path, *, window_name: str = "Candidate Labeler") -> int:
    crops = load_crops(session_dir)
    if not crops:
        raise ValueError(f"No candidate crops found in {session_dir / 'crops'}")

    store = LabelStore(session_dir / "labels.json")
    index = next((i for i, crop in enumerate(crops) if store.get(crop.name) is None), 0)

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 1)

    try:
        while True:
            crop = crops[index]
            image = cv2.imread(str(crop))
            if image is None:
                raise RuntimeError(f"Could not read crop: {crop}")

            canvas = _draw_overlay(image, index, len(crops), crop, store.get(crop.name), store.counts())
            cv2.imshow(window_name, canvas)
            key = cv2.waitKeyEx(0)

            if key in (ord("q"), ord("Q"), 27):
                return len(store.labels)

            if key in LABEL_KEYS or key in {ord(c.upper()) for c in "bdgnuxs"}:
                label = LABEL_KEYS.get(key)
                if label is None:
                    label = LABEL_KEYS.get(key + 32)
                if label is not None:
                    store.set(crop.name, label)
                    if index < len(crops) - 1:
                        index += 1
                    continue

            if key in (81, 2424832):  # left arrow on common OpenCV backends
                index = max(0, index - 1)
                continue

            if key in (83, 2555904):  # right arrow on common OpenCV backends
                index = min(len(crops) - 1, index + 1)
                continue
    finally:
        cv2.destroyWindow(window_name)


def main() -> None:
    parser = argparse.ArgumentParser(description="Human-label candidate crops from a collector session.")
    parser.add_argument("session", type=Path, help="training/raw/session_YYYYMMDD_HHMMSS")
    args = parser.parse_args()

    labeled = label_session(args.session)
    print(f"Labeling saved: {args.session / 'labels.json'}")
    print(f"Labeled/skipped entries: {labeled}")


if __name__ == "__main__":
    main()
