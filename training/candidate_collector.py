from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np


class CandidateCollector:
    """Persist screen-only detector candidates for later human labeling.

    This class is deliberately input-free: it only receives frames and the
    detector output. It never imports or calls any mouse/keyboard backend.
    """

    def __init__(
        self,
        root: str | Path = "training/raw",
        *,
        session_name: str | None = None,
        padding_px: int = 18,
        min_repeat_frames: int = 20,
        min_move_px: float = 12.0,
        max_saved_per_key: int = 500,
        click_crop_size_px: int = 96,
        write_metadata: bool = True,
    ) -> None:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        self.session_name = session_name or f"session_{timestamp}"
        self.session_dir = Path(root) / self.session_name
        self.crop_dir = self.session_dir / "crops"
        self.crop_dir.mkdir(parents=True, exist_ok=True)
        self.write_metadata = write_metadata
        self.metadata_path = self.session_dir / "metadata.jsonl"
        self._metadata_file = (
            self.metadata_path.open("a", encoding="utf-8")
            if write_metadata
            else None
        )
        self.padding_px = padding_px
        self.min_repeat_frames = min_repeat_frames
        self.min_move_px = min_move_px
        self.max_saved_per_key = max_saved_per_key
        self.click_crop_size_px = click_crop_size_px
        self._click_count = 0
        self._last_seen: list[dict[str, Any]] = []
        self._saved_counts: dict[str, int] = {}
        self.saved_count = 0

    @staticmethod
    def _center(bbox: tuple[int, int, int, int]) -> tuple[float, float]:
        x, y, w, h = bbox
        return x + w / 2.0, y + h / 2.0

    @staticmethod
    def _bbox_tuple(value: Any) -> tuple[int, int, int, int]:
        return tuple(int(round(float(v))) for v in value[:4])  # type: ignore[return-value]

    def _is_duplicate(
        self,
        *,
        frame_id: int,
        color: str,
        bbox: tuple[int, int, int, int],
        target_type: str | None,
    ) -> bool:
        cx, cy = self._center(bbox)
        _, _, w, h = bbox
        key = f"{color}:{target_type or 'unknown'}"
        for item in self._last_seen:
            if item["key"] != key:
                continue
            if frame_id - item["frame_id"] < self.min_repeat_frames:
                return True
            old_w, old_h = item["size"]
            old_cx, old_cy = item["center"]
            displacement = math.hypot(cx - old_cx, cy - old_cy)
            size_ratio = max(
                w / max(1.0, old_w),
                old_w / max(1.0, w),
                h / max(1.0, old_h),
                old_h / max(1.0, h),
            )
            if displacement < self.min_move_px and size_ratio < 1.25:
                return True
        return False

    def _remember(self, frame_id: int, key: str, bbox: tuple[int, int, int, int]) -> None:
        cx, cy = self._center(bbox)
        self._last_seen.append(
            {
                "frame_id": frame_id,
                "key": key,
                "center": (cx, cy),
                "size": (bbox[2], bbox[3]),
            }
        )
        if len(self._last_seen) > 256:
            self._last_seen = self._last_seen[-256:]

    @staticmethod
    def _json_safe(value: Any) -> Any:
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, dict):
            return {str(k): CandidateCollector._json_safe(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [CandidateCollector._json_safe(v) for v in value]
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        return str(value)

    def collect_frame(
        self,
        frame: np.ndarray,
        detections: list[dict],
        *,
        frame_id: int,
        capture_timestamp: float,
    ) -> int:
        """Save useful candidate crops from one detector frame.

        The collector samples each persistent candidate rather than dumping
        every frame. A substantially moved/resized candidate is eligible again.
        """
        height, width = frame.shape[:2]
        saved = 0

        for index, detection in enumerate(detections):
            raw_bbox = detection.get("bbox")
            if not raw_bbox or len(raw_bbox) < 4:
                continue
            x, y, w, h = self._bbox_tuple(raw_bbox)
            if w <= 0 or h <= 0:
                continue

            x0 = max(0, x - self.padding_px)
            y0 = max(0, y - self.padding_px)
            x1 = min(width, x + w + self.padding_px)
            y1 = min(height, y + h + self.padding_px)
            if x1 <= x0 or y1 <= y0:
                continue

            color = str(detection.get("color_name") or detection.get("color") or "unknown")
            target_type = detection.get("target_type")
            if hasattr(target_type, "value"):
                target_type = target_type.value
            target_type = str(target_type) if target_type is not None else None
            key = f"{color}:{target_type or 'unknown'}"

            if self._is_duplicate(
                frame_id=frame_id,
                color=color,
                bbox=(x, y, w, h),
                target_type=target_type,
            ):
                continue

            if self._saved_counts.get(key, 0) >= self.max_saved_per_key:
                continue

            crop = frame[y0:y1, x0:x1]
            filename = f"{frame_id:07d}_{index:03d}_{x}_{y}_{w}x{h}.png"
            crop_path = self.crop_dir / filename
            if not cv2.imwrite(str(crop_path), crop):
                continue

            metadata = {
                "session": self.session_name,
                "frame_id": frame_id,
                "capture_timestamp": capture_timestamp,
                "crop": str(crop_path.relative_to(self.session_dir)),
                "bbox": [x, y, w, h],
                "padded_bbox": [x0, y0, x1 - x0, y1 - y0],
                "color": color,
                "target_type": target_type,
                "candidate_index": index,
                "detection": self._json_safe(detection),
            }
            if self._metadata_file is not None:
                self._metadata_file.write(json.dumps(metadata, separators=(",", ":")) + "\n")
                self._metadata_file.flush()

            self._saved_counts[key] = self._saved_counts.get(key, 0) + 1
            self._remember(frame_id, key, (x, y, w, h))
            self.saved_count += 1
            saved += 1

        return saved

    def collect_click(
        self,
        frame: np.ndarray,
        *,
        screen_x: int,
        screen_y: int,
        region_left: int,
        region_top: int,
        frame_id: int,
    ) -> bool:
        """Save one crop for every click action, with no deduplication.

        The crop is centered on the actual screen click coordinate and taken
        from the frame already being processed. This keeps recording out of
        the input path: no extra screen capture is performed and misses are
        recorded exactly like hits.
        """
        height, width = frame.shape[:2]
        frame_x = int(screen_x) - int(region_left)
        frame_y = int(screen_y) - int(region_top)
        half = max(1, self.click_crop_size_px // 2)

        x0 = max(0, frame_x - half)
        y0 = max(0, frame_y - half)
        x1 = min(width, frame_x + half)
        y1 = min(height, frame_y + half)
        if x1 <= x0 or y1 <= y0:
            return False

        crop = frame[y0:y1, x0:x1]
        self._click_count += 1
        filename = (
            f"click_{self._click_count:07d}_f{frame_id:07d}_"
            f"{frame_x}_{frame_y}.png"
        )
        crop_path = self.crop_dir / filename
        if not cv2.imwrite(str(crop_path), crop):
            self._click_count -= 1
            return False
        self.saved_count += 1
        return True

    def close(self) -> None:
        if self._metadata_file is not None:
            self._metadata_file.close()

    def __enter__(self) -> "CandidateCollector":
        return self

    def __exit__(self, *_args: Any) -> None:
        self.close()
