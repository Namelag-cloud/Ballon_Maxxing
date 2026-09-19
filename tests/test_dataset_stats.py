from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from training.dataset_stats import dataset_counts, discover_label_files


class TestDatasetStats(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name) / "raw"
        self.root.mkdir()

    def tearDown(self):
        self.temp_dir.cleanup()

    def write_labels(self, session: str, labels: dict[str, str]) -> None:
        session_dir = self.root / session
        session_dir.mkdir()
        (session_dir / "labels.json").write_text(
            json.dumps(labels), encoding="utf-8"
        )

    def test_counts_all_requested_classes_skips_and_total(self):
        self.write_labels(
            "session_1",
            {
                "a.png": "balloon",
                "b.png": "duo",
                "c.png": "super_g_ring",
                "d.png": "notification",
                "e.png": "ui",
                "f.png": "unknown",
                "g.png": "skip",
            },
        )
        self.write_labels(
            "session_2",
            {
                "h.png": "balloon",
                "i.png": "balloon",
                "j.png": "skip",
            },
        )

        self.assertEqual(
            dataset_counts(self.root),
            {
                "balloon": 3,
                "duo": 1,
                "super_g_ring": 1,
                "notification": 1,
                "ui": 1,
                "unknown": 1,
                "skipped": 2,
                "total": 10,
            },
        )

    def test_ignores_non_session_root_entries(self):
        self.write_labels("session_1", {"a.png": "balloon"})
        (self.root / "README.txt").write_text("ignore", encoding="utf-8")
        (self.root / "not_a_session").mkdir()

        self.assertEqual(discover_label_files(self.root), [
            self.root / "session_1" / "labels.json"
        ])

    def test_rejects_unknown_label(self):
        self.write_labels("session_1", {"a.png": "banana"})
        with self.assertRaises(ValueError):
            dataset_counts(self.root)

    def test_missing_root_fails(self):
        with self.assertRaises(FileNotFoundError):
            dataset_counts(self.root / "missing")


if __name__ == "__main__":
    unittest.main()
