from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from config.settings import Settings
from vision.detector import Detection


@dataclass
class BalloonTrack:
    id: int
    position: tuple[int, int]
    bbox: tuple[int, int, int, int]
    color: str
    has_outline: bool
    confidence: float
    first_seen: float
    last_seen: float
    clicked: bool = False
    click_attempts: int = 0
    frames_missing: int = 0
    tracking_class: str | None = None
    metadata: dict = field(default_factory=dict)
    age_frames: int = 1
    total_displacement_px: float = 0.0
    mean_displacement_px: float = 0.0
    stationary_frames: int = 0
    max_bbox_delta_px: float = 0.0


class Tracker:
    def __init__(self, settings: Settings) -> None:
        self._config = settings.tracking
        self._tracks: dict[int, BalloonTrack] = {}
        self._next_id: int = 0
        self._frame_count: int = 0

    @property
    def tracks(self) -> dict[int, BalloonTrack]:
        return self._tracks

    def update(self, detections: list[Detection]) -> dict[int, BalloonTrack]:
        now = time.perf_counter_ns() / 1e9
        self._frame_count += 1

        for track in self._tracks.values():
            track.frames_missing += 1

        matched_detection_indices: set[int] = set()
        matched_track_ids: set[int] = set()

        if self._tracks and detections:
            dist_matrix = np.full((len(self._tracks), len(detections)), fill_value=1e9)
            track_ids = list(self._tracks.keys())

            for i, track_id in enumerate(track_ids):
                track = self._tracks[track_id]
                for j, det in enumerate(detections):
                    if track.tracking_class is not None and det.tracking_class is not None and track.tracking_class != det.tracking_class:
                        dist_matrix[i, j] = 1e9
                        continue
                    dx = track.position[0] - det.center[0]
                    dy = track.position[1] - det.center[1]
                    dist = (dx * dx + dy * dy) ** 0.5
                    if dist <= self._config.max_distance:
                        dist_matrix[i, j] = dist

            for _ in range(min(len(track_ids), len(detections))):
                min_idx = np.unravel_index(np.argmin(dist_matrix), dist_matrix.shape)
                min_dist = dist_matrix[min_idx]
                if min_dist >= 1e9:
                    break

                i, j = min_idx
                track_id = track_ids[i]
                det = detections[j]

                self._update_track(track_id, det, now)
                matched_detection_indices.add(j)
                matched_track_ids.add(track_id)

                dist_matrix[i, :] = 1e9
                dist_matrix[:, j] = 1e9

        for j, det in enumerate(detections):
            if j not in matched_detection_indices:
                self._create_track(det, now)

        stale_ids = [
            tid for tid, t in self._tracks.items()
            if t.frames_missing >= self._config.max_frames_missing
        ]
        for tid in stale_ids:
            del self._tracks[tid]

        return dict(self._tracks)

    def _create_track(self, det: Detection, now: float) -> None:
        track_id = self._next_id
        self._next_id += 1
        self._tracks[track_id] = BalloonTrack(
            id=track_id,
            position=det.center,
            bbox=det.bbox,
            color=det.color_name,
            has_outline=det.outline.has_outline,
            confidence=det.color_confidence,
            first_seen=now,
            last_seen=now,
            tracking_class=det.tracking_class,
            metadata={
                **dict(det.metadata),
                "click_attempts": 0,
                "temporal": {
                    "age_frames": 1,
                    "total_displacement_px": 0.0,
                    "mean_displacement_px": 0.0,
                    "stationary_frames": 0,
                    "max_bbox_delta_px": 0.0,
                    "static_ui_suspect": False,
                },
            },
            age_frames=1,
        )

    def mark_clicked(self, track_id: int) -> bool:
        """Record an actual click attempt for a tracked target."""
        track = self._tracks.get(track_id)
        if track is None or track.tracking_class != "normal_balloon":
            return False
        track.clicked = True
        track.click_attempts += 1
        track.metadata = {
            **dict(track.metadata),
            "click_attempts": track.click_attempts,
        }
        return True

    def _update_track(self, track_id: int, det: Detection, now: float) -> None:
        track = self._tracks[track_id]
        previous_position = track.position
        previous_bbox = track.bbox
        displacement = float(np.hypot(
            det.center[0] - previous_position[0],
            det.center[1] - previous_position[1],
        ))
        bbox_delta = max(
            abs(det.bbox[0] - previous_bbox[0]),
            abs(det.bbox[1] - previous_bbox[1]),
            abs(det.bbox[2] - previous_bbox[2]),
            abs(det.bbox[3] - previous_bbox[3]),
        )

        track.position = det.center
        track.bbox = det.bbox
        track.color = det.color_name
        track.has_outline = det.outline.has_outline
        track.confidence = det.color_confidence
        track.last_seen = now
        track.frames_missing = 0
        track.tracking_class = det.tracking_class
        track.age_frames += 1
        track.total_displacement_px += displacement
        track.mean_displacement_px = track.total_displacement_px / max(1, track.age_frames - 1)
        track.stationary_frames = track.stationary_frames + 1 if displacement < 1.0 else 0
        track.max_bbox_delta_px = max(track.max_bbox_delta_px, float(bbox_delta))

        static_ui_suspect = (
            track.age_frames >= 12
            and track.stationary_frames >= 10
            and track.mean_displacement_px < 0.75
            and track.max_bbox_delta_px <= 2.0
        )
        track.metadata = {
            **dict(det.metadata),
            "click_attempts": track.click_attempts,
            "temporal": {
                "age_frames": track.age_frames,
                "total_displacement_px": track.total_displacement_px,
                "mean_displacement_px": track.mean_displacement_px,
                "stationary_frames": track.stationary_frames,
                "max_bbox_delta_px": track.max_bbox_delta_px,
                "static_ui_suspect": static_ui_suspect,
            },
        }
