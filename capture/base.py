from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np

from config.settings import CaptureConfig


class CaptureBackend(ABC):
    @abstractmethod
    def __init__(self, config: CaptureConfig) -> None: ...

    @abstractmethod
    def grab(self) -> tuple[np.ndarray, float]:
        """Grab a frame and return (bgr_array, timestamp_monotonic)."""
        ...

    @abstractmethod
    def close(self) -> None: ...

    def __enter__(self) -> CaptureBackend:
        return self

    def __exit__(self, *args) -> None:
        self.close()
