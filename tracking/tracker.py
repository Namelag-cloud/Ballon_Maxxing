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
    frames_missing: int = 0


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
        )

    def _update_track(self, track_id: int, det: Detection, now: float) -> None:
        track = self._tracks[track_id]
        track.position = det.center
        track.bbox = det.bbox
        track.color = det.color_name
        track.has_outline = det.outline.has_outline
        track.confidence = det.color_confidence
        track.last_seen = now
        track.frames_missing = 0
