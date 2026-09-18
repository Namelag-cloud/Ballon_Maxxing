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

    @property
    def frame_log_duration_ms(self) -> float:
        return self._frame_log_duration_ms

    def begin_frame(self) -> None:
        self._frame_log_duration_ms = 0.0

    def log(self, event: str, **data: Any) -> None:
        t0 = time.perf_counter()
        try:
            self._seq += 1
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
        if not self._stream.closed:
            self._stream.close()

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
