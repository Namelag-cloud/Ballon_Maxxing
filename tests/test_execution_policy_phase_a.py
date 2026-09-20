from __future__ import annotations

import unittest

from behavior.balloon_executor import BalloonExecutor
from behavior.execution_policy import ExecutionContext
from behavior.fast_policy import FastExecutionPolicy
from config.settings import RunMode, Settings
from decision.mechanics import BalloonDecisionEngine
from game_profiles.balloon_types import BalloonColor, TargetType

from tests.test_balloon_execution_integration import FakeBehavior, FakeClock, FakeInput, balloon_action


class TestFastExecutionPolicy(unittest.TestCase):
    def setUp(self) -> None:
        self.settings = Settings()
        self.settings.mode = RunMode.LIVE
        self.settings.grouping.max_pair_distance = 200
        self.engine = BalloonDecisionEngine(self.settings)
        self.clock = FakeClock()
        self.input = FakeInput(self.clock)

    def _context(self, *, input_enabled: bool = True) -> ExecutionContext:
        return ExecutionContext(
            frame_id=1,
            now=self.clock(),
            capture_timestamp=0.0,
            region_left=0,
            region_top=0,
            input_enabled=input_enabled,
            duration_s=60.0,
            current_mouse_position=(0, 0),
        )

    def test_fast_policy_dispatches_current_visible_actions(self):
        policy = FastExecutionPolicy(clock=self.clock)
        decision = self.engine.decide([
            balloon_action(
                TargetType.NORMAL_BALLOON,
                BalloonColor.BLUE,
                (100, 100),
                50,
            ),
            balloon_action(
                TargetType.NORMAL_BALLOON,
                BalloonColor.BLUE,
                (200, 200),
                51,
            ),
        ])

        result = policy.tick(decision, self._context(), self.input)

        self.assertEqual(result.clicks_sent, 2)
        # Fast policy applies the configured humanization offset to normal clicks.
        expected = [(100, 100), (200, 200)]
        actual = [(x, y) for _, x, y in self.input.moves]
        self.assertEqual(len(actual), len(expected))
        for (x, y), (expected_x, expected_y) in zip(actual, expected):
            self.assertLessEqual(abs(x - expected_x), 2)
            self.assertLessEqual(abs(y - expected_y), 2)
        self.assertEqual(self.input.downs, [0.0, 0.0])

if __name__ == "__main__":
    unittest.main()
