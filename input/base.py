from __future__ import annotations

from abc import ABC, abstractmethod


class InputBackend(ABC):
    @abstractmethod
    def click(self, x: int, y: int) -> float:
        """Click at (x, y). Returns timestamp when click was sent."""
        ...

    @abstractmethod
    def close(self) -> None: ...

    def __enter__(self) -> InputBackend:
        return self

    def __exit__(self, *args) -> None:
        self.close()
