from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from training.dataset_split import Sample, session_audit, split_sessions, write_manifests


class TestDatasetSplit(unittest.TestCase):
    def make_samples(self):
        samples = []
        for session in range(8):
            name = f"session_{session}"
            samples.extend(
                Sample(name, f"{name}/a.png", "balloon") for _ in range(20)
            )
            samples.extend(
                Sample(name, f"{name}/u.png", "ui") for _ in range(4)
            )
            if session % 2 == 0:
                samples.append(Sample(name, f"{name}/d.png", "duo"))
        return samples

    def test_split_has_disjoint_sessions(self):
        splits = split_sessions(self.make_samples(), seed=7)
        session_sets = {
            name: {x.session for x in items} for name, items in splits.items()
        }
        self.assertTrue(session_sets["train"].isdisjoint(session_sets["val"]))
        self.assertTrue(session_sets["train"].isdisjoint(session_sets["test"]))
        self.assertTrue(session_sets["val"].isdisjoint(session_sets["test"]))
        self.assertEqual(set().union(*session_sets.values()), {f"session_{i}" for i in range(8)})

    def test_audit_reports_session_and_class_counts(self):
        audit = session_audit(self.make_samples())
        self.assertEqual(audit["sessions"], 8)
        self.assertEqual(audit["class_counts"]["balloon"], 160)
        self.assertEqual(audit["class_counts"]["duo"], 4)

    def test_manifests_are_jsonl(self):
        splits = split_sessions(self.make_samples(), seed=7)
        with tempfile.TemporaryDirectory() as tmp:
            summary = write_manifests(splits, Path(tmp), seed=7)
            self.assertEqual(set(summary["splits"]), {"train", "val", "test"})
            for split in ("train", "val", "test"):
                lines = (Path(tmp) / f"{split}.jsonl").read_text(encoding="utf-8").splitlines()
                self.assertEqual(len(lines), len(splits[split]))
                json.loads(lines[0])


if __name__ == "__main__":
    unittest.main()
