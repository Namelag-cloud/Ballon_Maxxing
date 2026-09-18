from __future__ import annotations

import ctypes
import time
from ctypes import wintypes
from typing import Any

from input.base import InputBackend
from input.coordinates import VirtualScreen, screen_to_sendinput

user32 = ctypes.windll.user32

INPUT_MOUSE = 0
MOUSEEVENTF_MOVE = 0x0001
MOUSEEVENTF_LEFTDOWN = 0x0002
MOUSEEVENTF_LEFTUP = 0x0004
MOUSEEVENTF_ABSOLUTE = 0x8000


class MOUSEINPUT(ctypes.Structure):
    _fields_ = [
        ("dx", ctypes.c_long), ("dy", ctypes.c_long),
        ("mouseData", ctypes.c_ulong), ("dwFlags", ctypes.c_ulong),
        ("time", ctypes.c_ulong), ("dwExtraInfo", ctypes.POINTER(ctypes.c_ulong)),
    ]


class INPUT(ctypes.Structure):
    class _INPUT_UNION(ctypes.Union):
        _fields_ = [("mi", MOUSEINPUT)]
    _fields_ = [("type", ctypes.c_ulong), ("union", _INPUT_UNION)]


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
    def __init__(self, debug_logging: bool = False, diagnostic: Any = None) -> None:
        self._virtual = VirtualScreen()
        self._debug_logging = debug_logging
        self._diagnostic = diagnostic
        self._last_x = self._virtual.left + self._virtual.width // 2
        self._last_y = self._virtual.top + self._virtual.height // 2

    @property
    def virtual_screen(self) -> VirtualScreen:
        return self._virtual

    def click_screen(self, screen_x: int, screen_y: int) -> float:
        self.move_to(screen_x, screen_y)
        self.left_down()
        return self.left_up()

    def move_to(self, screen_x: int, screen_y: int) -> float:
        norm_x, norm_y = screen_to_sendinput(
            screen_x, screen_y, self._virtual.left, self._virtual.top,
            self._virtual.width, self._virtual.height,
        )
        move = _make_input(norm_x, norm_y, MOUSEEVENTF_MOVE | MOUSEEVENTF_ABSOLUTE)
        inputs = (INPUT * 1)(move)
        sent = user32.SendInput(1, ctypes.byref(inputs), ctypes.sizeof(INPUT))
        ts = time.perf_counter_ns() / 1e9
        if sent != 1:
            if self._diagnostic:
                self._diagnostic.log("sendinput_dispatch_error", kind="move", x=screen_x, y=screen_y, sent=sent)
            raise RuntimeError(f"SendInput move dispatch incomplete: sent {sent}/1")
        self._last_x, self._last_y = int(screen_x), int(screen_y)
        if self._diagnostic:
            self._diagnostic.log("sendinput_move", x=screen_x, y=screen_y, sent=sent)
        return ts

    def left_down(self) -> float:
        down = _make_input(0, 0, MOUSEEVENTF_LEFTDOWN)
        inputs = (INPUT * 1)(down)
        sent = user32.SendInput(1, ctypes.byref(inputs), ctypes.sizeof(INPUT))
        ts = time.perf_counter_ns() / 1e9
        if sent != 1:
            if self._diagnostic:
                self._diagnostic.log("sendinput_dispatch_error", kind="mouse_down", sent=sent)
            raise RuntimeError(f"SendInput left-down dispatch incomplete: sent {sent}/1")
        if self._diagnostic:
            self._diagnostic.log("sendinput_mouse_down", sent=sent, x=self._last_x, y=self._last_y)
        return ts

    def left_up(self) -> float:
        up = _make_input(0, 0, MOUSEEVENTF_LEFTUP)
        inputs = (INPUT * 1)(up)
        sent = user32.SendInput(1, ctypes.byref(inputs), ctypes.sizeof(INPUT))
        ts = time.perf_counter_ns() / 1e9
        if sent != 1:
            if self._diagnostic:
                self._diagnostic.log("sendinput_dispatch_error", kind="mouse_up", sent=sent)
            raise RuntimeError(f"SendInput left-up dispatch incomplete: sent {sent}/1")
        if self._diagnostic:
            self._diagnostic.log("sendinput_mouse_up", sent=sent, x=self._last_x, y=self._last_y)
        return ts

    def click_pair(
        self,
        first: tuple[int, int],
        second: tuple[int, int],
        inter_click_delay_s: float = 0.0,
    ) -> tuple[float, float, float, float]:
        self.move_to(*first)
        first_down = self.left_down()
        first_up = self.left_up()

        if inter_click_delay_s > 0:
            time.sleep(inter_click_delay_s)

        self.move_to(*second)
        second_down = self.left_down()
        second_up = self.left_up()
        return first_down, first_up, second_down, second_up

    def click(self, x: int, y: int) -> float:
        return self.click_screen(x, y)

    def close(self) -> None:
        pass
