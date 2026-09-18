from __future__ import annotations

import csv
import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import numpy as np

from telemetry.timing import now_monotonic
from telemetry.behavioral import BehavioralExtractor, BehavioralFeatures
from telemetry.outcome import SessionOutcome


@dataclass
class FrameTelemetry:
    frame_id: int
    capture_timestamp: float
    detection_timestamp: float
    decision_timestamp: float
    input_timestamp: Optional[float]
    processing_duration: float
    end_to_end_latency: float
    detections_count: int
    tracks_count: int
    clicks_sent: int
    behavioral: BehavioralFeatures
    action_timestamps: Optional[ActionTimestamps] = None


class TelemetryLogger:
    def __init__(self, output_dir: str = "telemetry_output") -> None:
        self._output_dir = Path(output_dir)
        self._output_dir.mkdir(exist_ok=True)
        self._frames: list[FrameTelemetry] = []
        self._behavioral = BehavioralExtractor()
        self._frame_counter: int = 0
        self._total_clicks: int = 0
        self._outcomes: list[SessionOutcome] = []
        self._session_metadata: dict = {}

        ts = int(time.time())
        self._csv_path = self._output_dir / f"frames_{ts}.csv"
        self._json_path = self._output_dir / f"summary_{ts}.json"
        self._csv_file = open(self._csv_path, "w", newline="")
        self._csv_writer = csv.writer(self._csv_file)
        self._csv_writer.writerow([
            "frame_id", "capture_ts", "detection_ts", "decision_ts", "input_ts",
            "processing_ms", "e2e_ms", "detections", "tracks", "clicks",
            "action_planned", "reaction_start", "reaction_end",
            "movement_start", "movement_end",
            "click_down", "click_up", "action_complete",
        ])

    @property
    def behavioral(self) -> BehavioralExtractor:
        return self._behavioral

    def log_frame(self, telem: FrameTelemetry) -> None:
        self._frames.append(telem)
        ts = telem.action_timestamps
        self._csv_writer.writerow([
            telem.frame_id,
            f"{telem.capture_timestamp:.6f}",
            f"{telem.detection_timestamp:.6f}",
            f"{telem.decision_timestamp:.6f}",
            f"{telem.input_timestamp:.6f}" if telem.input_timestamp else "",
            f"{telem.processing_duration:.3f}",
            f"{telem.end_to_end_latency:.3f}",
            telem.detections_count,
            telem.tracks_count,
            telem.clicks_sent,
            f"{ts.action_planned:.6f}" if ts else "",
            f"{ts.reaction_start:.6f}" if ts else "",
            f"{ts.reaction_end:.6f}" if ts else "",
            f"{ts.movement_start:.6f}" if ts else "",
            f"{ts.movement_end:.6f}" if ts else "",
            f"{ts.click_down:.6f}" if ts else "",
            f"{ts.click_up:.6f}" if ts else "",
            f"{ts.action_complete:.6f}" if ts else "",
        ])
        self._csv_file.flush()

    def record_click(self, **kwargs) -> None:
        self._total_clicks += 1
        self._behavioral.record_click(**kwargs)

    def set_session_metadata(self, **metadata) -> None:
        self._session_metadata.update(metadata)

    def record_outcome(self, outcome: SessionOutcome) -> None:
        self._outcomes.append(outcome)

    def summary(self) -> dict:
        if not self._frames:
            return {"error": "no frames recorded"}

        durations = [f.processing_duration for f in self._frames]
        latencies = [f.end_to_end_latency for f in self._frames]
        detect_counts = [f.detections_count for f in self._frames]

        action_timestamps = [
            f.action_timestamps for f in self._frames if f.action_timestamps
        ]
        action_stats = {}
        if action_timestamps:
            reaction_times = [
                ts.reaction_end - ts.reaction_start for ts in action_timestamps
                if ts.reaction_end > 0 and ts.reaction_start > 0
            ]
            movement_times = [
                ts.movement_end - ts.movement_start for ts in action_timestamps
                if ts.movement_end > 0 and ts.movement_start > 0
            ]
            click_durations = [
                ts.click_up - ts.click_down for ts in action_timestamps
                if ts.click_up > 0 and ts.click_down > 0
            ]
            total_times = [
                ts.action_complete - ts.action_planned for ts in action_timestamps
                if ts.action_complete > 0 and ts.action_planned > 0
            ]
            action_stats = {
                "total_actions": len(action_timestamps),
                "reaction_time_ms": _stats_list(reaction_times, 1000) if reaction_times else None,
                "movement_time_ms": _stats_list(movement_times, 1000) if movement_times else None,
                "click_duration_ms": _stats_list(click_durations, 1000) if click_durations else None,
                "total_action_time_ms": _stats_list(total_times, 1000) if total_times else None,
            }

        return {
            "total_frames": len(self._frames),
            "total_clicks": self._total_clicks,
            "fps": len(self._frames) / max(0.001, self._frames[-1].end_to_end_latency),
            "processing_ms": {
                "mean": float(np.mean(durations)),
                "median": float(np.median(durations)),
                "p95": float(np.percentile(durations, 95)),
                "p99": float(np.percentile(durations, 99)),
                "min": float(np.min(durations)),
                "max": float(np.max(durations)),
            },
            "e2e_latency_ms": {
                "mean": float(np.mean(latencies)),
                "median": float(np.median(latencies)),
                "p95": float(np.percentile(latencies, 95)),
                "p99": float(np.percentile(latencies, 99)),
            },
            "avg_detections_per_frame": float(np.mean(detect_counts)),
            "action_timing": action_stats,
            "behavioral": asdict(self._behavioral.extract(len(detect_counts))),
            "session": self._session_metadata,
            "outcomes_count": len(self._outcomes),
            "outcomes": [o.as_dict() for o in self._outcomes],
        }

    def save(self) -> None:
        self._csv_file.close()
        summary = self.summary()
        with open(self._json_path, "w") as f:
            json.dump(summary, f, indent=2)

    def close(self) -> None:
        self.save()


def _stats_list(values: list[float], scale: float = 1.0) -> dict:
    arr = np.array(values) * scale
    return {
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
        "count": len(values),
    }
