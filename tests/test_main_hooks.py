from __future__ import annotations

import ctypes
import unittest

from main import _MouseHookStruct


class TestMouseHookStruct(unittest.TestCase):
    def test_matches_msllhookstruct_layout(self):
        fixed_prefix = (
            2 * ctypes.sizeof(ctypes.wintypes.LONG)
            + 3 * ctypes.sizeof(ctypes.wintypes.DWORD)
        )
        pointer_size = ctypes.sizeof(ctypes.c_void_p)
        expected_size = (
            (fixed_prefix + pointer_size - 1) // pointer_size
        ) * pointer_size + pointer_size
        self.assertEqual(ctypes.sizeof(_MouseHookStruct), expected_size)

        fields = {name for name, _ in _MouseHookStruct._fields_}
        self.assertEqual(
            fields,
            {"pt_x", "pt_y", "mouseData", "flags", "time", "dwExtraInfo"},
        )


if __name__ == "__main__":
    unittest.main()
