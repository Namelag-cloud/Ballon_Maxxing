from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from training.label_candidates import LabelStore, discover_sessions, load_crops


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
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_returns_supported_images_only(self):
        session_dir = self.tmp_path / "session"
        crop_dir = session_dir / "crops"
        crop_dir.mkdir(parents=True)
        (crop_dir / "b.png").write_bytes(b"")
        (crop_dir / "a.jpg").write_bytes(b"")
        (crop_dir / "notes.txt").write_text("ignore", encoding="utf-8")
        self.assertEqual(
            [path.name for path in load_crops(session_dir)],
            ["a.jpg", "b.png"],
        )


class TestDiscoverSessions(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_finds_only_directories_with_crops(self):
        root = self.tmp_path / "raw"
        root.mkdir()
        (root / "session_1" / "crops").mkdir(parents=True)
        (root / "session_2" / "crops").mkdir(parents=True)
        (root / "not_a_session").mkdir()
        (root / "file.txt").write_text("ignore", encoding="utf-8")

        self.assertEqual(
            [path.name for path in discover_sessions(root)],
            ["session_1", "session_2"],
        )


if __name__ == "__main__":
    unittest.main()
