from __future__ import annotations

import ctypes


_SM_XVIRTUALSCREEN = 76
_SM_YVIRTUALSCREEN = 77
_SM_CXVIRTUALSCREEN = 78
_SM_CYVIRTUALSCREEN = 79

_user32 = ctypes.windll.user32


class VirtualScreen:
    __slots__ = ("left", "top", "width", "height")

    def __init__(self) -> None:
        self.left: int = _user32.GetSystemMetrics(_SM_XVIRTUALSCREEN)
        self.top: int = _user32.GetSystemMetrics(_SM_YVIRTUALSCREEN)
        self.width: int = _user32.GetSystemMetrics(_SM_CXVIRTUALSCREEN)
        self.height: int = _user32.GetSystemMetrics(_SM_CYVIRTUALSCREEN)

    def __repr__(self) -> str:
        return f"VirtualScreen(left={self.left}, top={self.top}, width={self.width}, height={self.height})"


def frame_to_screen(
    frame_x: int,
    frame_y: int,
    region_left: int,
    region_top: int,
) -> tuple[int, int]:
    return (frame_x + region_left, frame_y + region_top)


def screen_to_sendinput(
    screen_x: int,
    screen_y: int,
    virtual_left: int,
    virtual_top: int,
    virtual_width: int,
    virtual_height: int,
) -> tuple[int, int]:
    if virtual_width <= 1:
        norm_x = 0
    else:
        norm_x = int((screen_x - virtual_left) * 65535 / (virtual_width - 1))

    if virtual_height <= 1:
        norm_y = 0
    else:
        norm_y = int((screen_y - virtual_top) * 65535 / (virtual_height - 1))

    norm_x = max(0, min(65535, norm_x))
    norm_y = max(0, min(65535, norm_y))

    return (norm_x, norm_y)


def frame_to_sendinput(
    frame_x: int,
    frame_y: int,
    region_left: int,
    region_top: int,
    virtual_left: int,
    virtual_top: int,
    virtual_width: int,
    virtual_height: int,
) -> tuple[int, int]:
    screen_x, screen_y = frame_to_screen(frame_x, frame_y, region_left, region_top)
    return screen_to_sendinput(
        screen_x, screen_y,
        virtual_left, virtual_top,
        virtual_width, virtual_height,
    )


def diagnose(
    frame_x: int,
    frame_y: int,
    region_left: int,
    region_top: int,
    virtual: VirtualScreen,
    norm_x: int,
    norm_y: int,
) -> str:
    screen_x, screen_y = frame_to_screen(frame_x, frame_y, region_left, region_top)
    return (
        f"frame=({frame_x},{frame_y}) "
        f"region=(left={region_left},top={region_top}) "
        f"screen=({screen_x},{screen_y}) "
        f"virtual=({virtual.left},{virtual.top},{virtual.width}x{virtual.height}) "
        f"sendinput=({norm_x},{norm_y})"
    )
