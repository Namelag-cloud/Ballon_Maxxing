from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from training.label_candidates import LabelStore, load_crops


class TestLabelStore(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_persists_and_resumes(self):
        path = self.tmp_path / "labels.json"
        store = LabelStore(path)
        store.set("one.png", "balloon")
        store.set("two.png", "ui")
        store.set("three.png", "duo")
        store.set("four.png", "super_g_ring")
        store.set("five.png", "skip")

        resumed = LabelStore(path)
        self.assertEqual(resumed.get("one.png"), "balloon")
        self.assertEqual(resumed.get("two.png"), "ui")
        self.assertEqual(resumed.get("three.png"), "duo")
        self.assertEqual(resumed.get("four.png"), "super_g_ring")
        self.assertEqual(resumed.get("five.png"), "skip")
        self.assertEqual(
            resumed.counts(),
            {
                "balloon": 1,
                "ui": 1,
                "notification": 0,
                "duo": 1,
                "super_g_ring": 1,
                "unknown": 0,
                "skip": 1,
            },
        )

    def test_rejects_unknown_labels(self):
        path = self.tmp_path / "labels.json"
        path.write_text(json.dumps({"one.png": "banana"}), encoding="utf-8")
        with self.assertRaises(ValueError):
            LabelStore(path)

    def test_rejects_setting_unknown_label(self):
        store = LabelStore(self.tmp_path / "labels.json")
        with self.assertRaises(ValueError):
            store.set("one.png", "banana")


class TestLoadCrops(unittest.TestCase):
    def test_returns_supported_images_only(self):
        crop_dir = Path(tempfile.mkdtemp(dir=self.tmp_path))
        (crop_dir / "b.png").write_bytes(b"")
        (crop_dir / "a.jpg").write_bytes(b"")
        (crop_dir / "notes.txt").write_text("ignore", encoding="utf-8")
        self.assertEqual(
            [path.name for path in load_crops(crop_dir.parent)],
            ["a.jpg", "b.png"],
        )


if __name__ == "__main__":
    unittest.main()
