from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from diagnostics.downstream import DownstreamDiagnostic


class DownstreamDiagnosticTests(unittest.TestCase):
    def test_records_click_followup_and_session_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "trace.jsonl"
            diagnostic = DownstreamDiagnostic(path)

            diagnostic.log(
                "click_context",
                target_id=7,
                decision_to_click_ms=2.5,
            )
            diagnostic.log(
                "click_followup",
                target_id=7,
                status="absent_next_frame",
            )
            diagnostic.log(
                "decision_state_transition",
                accepted=True,
            )
            diagnostic.close()

            records = [
                json.loads(line)
                for line in path.read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual([r["event"] for r in records], [
                "click_context",
                "click_followup",
                "decision_state_transition",
                "diagnostic_summary",
            ])
            summary = records[-1]
            self.assertEqual(summary["click_followup_status_counts"], {
                "absent_next_frame": 1,
            })
            self.assertEqual(summary["state_transition_count"], 1)
            self.assertEqual(summary["click_decision_latency_ms"]["count"], 1)
            self.assertAlmostEqual(
                summary["click_decision_latency_ms"]["median"],
                2.5,
            )
            self.assertEqual(summary["event_counts"]["click_context"], 1)

    def test_close_is_idempotent(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "trace.jsonl"
            diagnostic = DownstreamDiagnostic(path)
            diagnostic.close()
            diagnostic.close()
            self.assertEqual(len(path.read_text(encoding="utf-8").splitlines()), 1)


if __name__ == "__main__":
    unittest.main()
