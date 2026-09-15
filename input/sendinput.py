from __future__ import annotations

import ctypes
import time
from ctypes import wintypes

import numpy as np

from input.base import InputBackend

user32 = ctypes.windll.user32

INPUT_MOUSE = 0
MOUSEEVENTF_MOVE = 0x0001
MOUSEEVENTF_LEFTDOWN = 0x0002
MOUSEEVENTF_LEFTUP = 0x0004
MOUSEEVENTF_ABSOLUTE = 0x8000

SCREEN_W = user32.GetSystemMetrics(0)
SCREEN_H = user32.GetSystemMetrics(1)


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
    def __init__(self) -> None:
        pass

    def click(self, x: int, y: int) -> float:
        abs_x = int(x * 65535 / SCREEN_W)
        abs_y = int(y * 65535 / SCREEN_H)

        move = _make_input(abs_x, abs_y, MOUSEEVENTF_MOVE | MOUSEEVENTF_ABSOLUTE)
        down = _make_input(0, 0, MOUSEEVENTF_LEFTDOWN)
        up = _make_input(0, 0, MOUSEEVENTF_LEFTUP)

        inputs = (INPUT * 3)(move, down, up)
        user32.SendInput(3, ctypes.byref(inputs), ctypes.sizeof(INPUT))

        return time.perf_counter_ns() / 1e9

    def close(self) -> None:
        pass
