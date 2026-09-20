from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any


class DownstreamDiagnostic:
    """Temporary JSONL trace for the balloon downstream execution lifecycle."""

    def __init__(self, path: str | os.PathLike[str] = "diagnostics/downstream_trace.jsonl") -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._stream = self.path.open("w", encoding="utf-8")
        self._seq = 0
        self._frame_log_duration_ms = 0.0
        self._event_counts: dict[str, int] = {}
        self._click_decision_latencies_ms: list[float] = []
        self._click_to_next_capture_ms: list[float] = []
        self._click_to_next_decision_ms: list[float] = []
        self._followup_status_counts: dict[str, int] = {}
        self._state_transition_count = 0
        self._click_timing_samples: list[dict[str, Any]] = []

    @property
    def frame_log_duration_ms(self) -> float:
        return self._frame_log_duration_ms

    def begin_frame(self) -> None:
        self._frame_log_duration_ms = 0.0

    def log(self, event: str, **data: Any) -> None:
        t0 = time.perf_counter()
        try:
            self._seq += 1
            self._event_counts[event] = self._event_counts.get(event, 0) + 1
            if event == "click_context":
                latency = data.get("decision_to_click_ms")
                if isinstance(latency, (int, float)):
                    self._click_decision_latencies_ms.append(float(latency))
            elif event == "click_followup":
                capture_latency = data.get("click_to_next_capture_ms")
                decision_latency = data.get("click_to_next_decision_ms")
                if isinstance(capture_latency, (int, float)):
                    self._click_to_next_capture_ms.append(float(capture_latency))
                if isinstance(decision_latency, (int, float)):
                    self._click_to_next_decision_ms.append(float(decision_latency))
                status = data.get("status")
                if status is not None:
                    self._followup_status_counts[str(status)] = self._followup_status_counts.get(str(status), 0) + 1
            elif event == "decision_state_transition":
                self._state_transition_count += 1
            elif event == "click_timing":
                self._click_timing_samples.append(dict(data))
            record = {
                "seq": self._seq,
                "t_mono": time.perf_counter(),
                "event": event,
                **self._jsonable(data),
            }
            self._stream.write(json.dumps(record, separators=(",", ":"), sort_keys=True) + "\n")
            self._stream.flush()
        finally:
            self._frame_log_duration_ms += (time.perf_counter() - t0) * 1000.0

    def close(self) -> None:
        if self._stream.closed:
            return
        self.log(
            "diagnostic_summary",
            event_counts=dict(self._event_counts),
            click_decision_latency_ms=self._stats(self._click_decision_latencies_ms),
            click_to_next_capture_ms=self._stats(self._click_to_next_capture_ms),
            click_to_next_decision_ms=self._stats(self._click_to_next_decision_ms),
            click_followup_status_counts=dict(self._followup_status_counts),
            click_execution_timing=self._click_timing_summary(),
            state_transition_count=self._state_transition_count,
        )
        self._stream.close()

    def _click_timing_summary(self) -> dict:
        def summarize(samples: list[dict[str, Any]]) -> dict:
            metric_names = (
                "movement_ms",
                "click_hold_ms",
                "input_total_ms",
                "inter_click_down_gap_ms",
            )
            result = {"count": len(samples)}
            for metric in metric_names:
                values = [
                    float(sample[metric])
                    for sample in samples
                    if isinstance(sample.get(metric), (int, float))
                ]
                result[metric] = self._stats(values)
            result["total_input_ms"] = sum(
                float(sample["input_total_ms"])
                for sample in samples
                if isinstance(sample.get("input_total_ms"), (int, float))
            )
            return result

        by_kind: dict[str, list[dict[str, Any]]] = {}
        by_lane: dict[str, list[dict[str, Any]]] = {}
        for sample in self._click_timing_samples:
            kind = str(sample.get("click_kind", "unknown"))
            lane = str(sample.get("lane", "unknown"))
            by_kind.setdefault(kind, []).append(sample)
            by_lane.setdefault(lane, []).append(sample)

        return {
            "overall": summarize(self._click_timing_samples),
            "by_click_kind": {
                key: summarize(value)
                for key, value in sorted(by_kind.items())
            },
            "by_lane": {
                key: summarize(value)
                for key, value in sorted(by_lane.items())
            },
        }

    @staticmethod
    def _stats(values: list[float]) -> dict:
        if not values:
            return {"count": 0}
        ordered = sorted(values)
        n = len(ordered)
        return {
            "count": n,
            "mean": sum(ordered) / n,
            "min": ordered[0],
            "median": ordered[n // 2] if n % 2 else (ordered[n // 2 - 1] + ordered[n // 2]) / 2,
            "max": ordered[-1],
        }

    @staticmethod
    def _jsonable(value: Any) -> Any:
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        if isinstance(value, dict):
            return {str(k): DownstreamDiagnostic._jsonable(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [DownstreamDiagnostic._jsonable(v) for v in value]
        if hasattr(value, "value"):
            return DownstreamDiagnostic._jsonable(value.value)
        if hasattr(value, "name") and value.__class__.__module__ == "enum":
            return value.name
        return str(value)
