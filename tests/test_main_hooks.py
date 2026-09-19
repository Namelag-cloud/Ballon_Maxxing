from __future__ import annotations

import ctypes
import unittest

from main import _MouseHookStruct


class TestMouseHookStruct(unittest.TestCase):
    def test_matches_msllhookstruct_layout(self):
        self.assertEqual(
            ctypes.sizeof(_MouseHookStruct),
            20 + ctypes.sizeof(ctypes.c_void_p),
        )

        fields = {name for name, _ in _MouseHookStruct._fields_}
        self.assertEqual(
            fields,
            {"pt_x", "pt_y", "mouseData", "flags", "time", "dwExtraInfo"},
        )


if __name__ == "__main__":
    unittest.main()
