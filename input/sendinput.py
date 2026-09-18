from __future__ import annotations

import ctypes
import time
from ctypes import wintypes

from input.base import InputBackend
from input.coordinates import VirtualScreen, screen_to_sendinput, diagnose

user32 = ctypes.windll.user32

INPUT_MOUSE = 0
MOUSEEVENTF_MOVE = 0x0001
MOUSEEVENTF_LEFTDOWN = 0x0002
MOUSEEVENTF_LEFTUP = 0x0004
MOUSEEVENTF_ABSOLUTE = 0x8000


class MOUSEINPUT(ctypes.Structure):
    _fields_ = [
        ("dx", ctypes.c_long),
        ("dy", ctypes.c_long),
        ("mouseData", ctypes.c_ulong),
        ("dwFlags", ctypes.c_ulong),
        ("time", ctypes.c_ulong),
        ("dwExtraInfo", ctypes.POINTER(ctypes.c_ulong)),
    ]


class INPUT(ctypes.Structure):
    class _INPUT_UNION(ctypes.Union):
        _fields_ = [("mi", MOUSEINPUT)]
    _fields_ = [
        ("type", ctypes.c_ulong),
        ("union", _INPUT_UNION),
    ]


def _make_input(dx: int, dy: int, flags: int) -> INPUT:
    inp = INPUT()
    inp.type = INPUT_MOUSE
    inp.union.mi.dx = dx
    inp.union.mi.dy = dy
    inp.union.mi.dwFlags = flags
    inp.union.mi.time = 0
    inp.union.mi.dwExtraInfo = ctypes.pointer(ctypes.c_ulong(0))
    return inp


class SendInputBackend(InputBackend):
    def __init__(self, debug_logging: bool = False) -> None:
        self._virtual = VirtualScreen()
        self._debug_logging = debug_logging
        self._last_x: int = self._virtual.left + self._virtual.width // 2
        self._last_y: int = self._virtual.top + self._virtual.height // 2

    @property
    def virtual_screen(self) -> VirtualScreen:
        return self._virtual

    def click_screen(self, screen_x: int, screen_y: int) -> float:
        norm_x, norm_y = screen_to_sendinput(
            screen_x, screen_y,
            self._virtual.left, self._virtual.top,
            self._virtual.width, self._virtual.height,
        )

        move = _make_input(norm_x, norm_y, MOUSEEVENTF_MOVE | MOUSEEVENTF_ABSOLUTE)
        down = _make_input(0, 0, MOUSEEVENTF_LEFTDOWN)
        up = _make_input(0, 0, MOUSEEVENTF_LEFTUP)

        inputs = (INPUT * 3)(move, down, up)
        sent = user32.SendInput(3, ctypes.byref(inputs), ctypes.sizeof(INPUT))
        if sent != 3:
            pass

        return time.perf_counter_ns() / 1e9

    def move_to(self, screen_x: int, screen_y: int) -> float:
        norm_x, norm_y = screen_to_sendinput(
            screen_x, screen_y,
            self._virtual.left, self._virtual.top,
            self._virtual.width, self._virtual.height,
        )
        move = _make_input(norm_x, norm_y, MOUSEEVENTF_MOVE | MOUSEEVENTF_ABSOLUTE)
        inputs = (INPUT * 1)(move)
        user32.SendInput(1, ctypes.byref(inputs), ctypes.sizeof(INPUT))
        return time.perf_counter_ns() / 1e9

    def left_down(self) -> float:
        down = _make_input(0, 0, MOUSEEVENTF_LEFTDOWN)
        inputs = (INPUT * 1)(down)
        user32.SendInput(1, ctypes.byref(inputs), ctypes.sizeof(INPUT))
        return time.perf_counter_ns() / 1e9

    def left_up(self) -> float:
        up = _make_input(0, 0, MOUSEEVENTF_LEFTUP)
        inputs = (INPUT * 1)(up)
        user32.SendInput(1, ctypes.byref(inputs), ctypes.sizeof(INPUT))
        return time.perf_counter_ns() / 1e9

    def click_pair(
        self,
        first: tuple[int, int],
        second: tuple[int, int],
    ) -> tuple[float, float, float, float]:
        """Dispatch two complete clicks in one SendInput batch.

        The returned timestamps are the authoritative dispatch timestamps used
        by the twin scheduler to enforce the <=30 ms mouse-down contract.
        """
        inputs_list = []
        for screen_x, screen_y in (first, second):
            norm_x, norm_y = screen_to_sendinput(
                screen_x, screen_y,
                self._virtual.left, self._virtual.top,
                self._virtual.width, self._virtual.height,
            )
            inputs_list.extend([
                _make_input(norm_x, norm_y, MOUSEEVENTF_MOVE | MOUSEEVENTF_ABSOLUTE),
                _make_input(0, 0, MOUSEEVENTF_LEFTDOWN),
                _make_input(0, 0, MOUSEEVENTF_LEFTUP),
            ])

        inputs = (INPUT * len(inputs_list))(*inputs_list)
        sent = user32.SendInput(
            len(inputs_list), ctypes.byref(inputs), ctypes.sizeof(INPUT)
        )
        if sent != len(inputs_list):
            raise RuntimeError(
                f"SendInput pair dispatch incomplete: sent {sent}/{len(inputs_list)} events"
            )

        timestamp = time.perf_counter_ns() / 1e9
        return timestamp, timestamp, timestamp, timestamp

    def click(self, x: int, y: int) -> float:
        return self.click_screen(x, y)

    def close(self) -> None:
        pass
