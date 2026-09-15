from __future__ import annotations

import csv
import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

from telemetry.timing import now_monotonic
from telemetry.behavioral import BehavioralExtractor, BehavioralFeatures


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


class TelemetryLogger:
    def __init__(self, output_dir: str = "telemetry_output") -> None:
        self._output_dir = Path(output_dir)
        self._output_dir.mkdir(exist_ok=True)
        self._frames: list[FrameTelemetry] = []
        self._behavioral = BehavioralExtractor()
        self._frame_counter: int = 0
        self._total_clicks: int = 0

        ts = int(time.time())
        self._csv_path = self._output_dir / f"frames_{ts}.csv"
        self._json_path = self._output_dir / f"summary_{ts}.json"
        self._csv_file = open(self._csv_path, "w", newline="")
        self._csv_writer = csv.writer(self._csv_file)
        self._csv_writer.writerow([
            "frame_id", "capture_ts", "detection_ts", "decision_ts", "input_ts",
            "processing_ms", "e2e_ms", "detections", "tracks", "clicks",
        ])

    @property
    def behavioral(self) -> BehavioralExtractor:
        return self._behavioral

    def log_frame(self, telem: FrameTelemetry) -> None:
        self._frames.append(telem)
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
        ])
        self._csv_file.flush()

    def record_click(self, **kwargs) -> None:
        self._total_clicks += 1
        self._behavioral.record_click(**kwargs)

    def summary(self) -> dict:
        if not self._frames:
            return {"error": "no frames recorded"}

        durations = [f.processing_duration for f in self._frames]
        latencies = [f.end_to_end_latency for f in self._frames]
        detect_counts = [f.detections_count for f in self._frames]

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
            "behavioral": asdict(self._behavioral.extract(len(detect_counts))),
        }

    def save(self) -> None:
        self._csv_file.close()
        summary = self.summary()
        with open(self._json_path, "w") as f:
            json.dump(summary, f, indent=2)

    def close(self) -> None:
        self.save()


import numpy as np
