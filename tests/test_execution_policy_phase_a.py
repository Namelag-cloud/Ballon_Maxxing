from __future__ import annotations

import unittest
from types import SimpleNamespace

from behavior.balloon_executor import BalloonExecutor
from behavior.execution_policy import ExecutionContext
from behavior.fast_policy import FastExecutionPolicy
from behavior.humanized_policy import HumanizedExecutionPolicy
from config.settings import RunMode, Settings
from decision.mechanics import BalloonDecisionEngine
from game_profiles.balloon_types import BalloonColor, TargetType
from main import parse_args, resolve_policy_name

from tests.test_balloon_execution_integration import FakeBehavior, FakeClock, FakeInput, balloon_action


class TestExecutionPolicyPhaseA(unittest.TestCase):
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

    def test_cli_policy_values_and_precedence(self):
        self.assertIsNone(parse_args([]).policy)
        self.assertEqual(parse_args(["--policy", "fast"]).policy, "fast")
        self.assertEqual(
            parse_args(["--policy", "humanized"]).policy,
            "humanized",
        )

        base = SimpleNamespace(policy=None, fast=False)
        self.assertEqual(
            resolve_policy_name(base, self.settings, None),
            "fast",
        )

        fast_flag = SimpleNamespace(policy=None, fast=True)
        self.assertEqual(
            resolve_policy_name(fast_flag, self.settings, None),
            "fast",
        )

        explicit_humanized = SimpleNamespace(policy="humanized", fast=True)
        self.assertEqual(
            resolve_policy_name(explicit_humanized, self.settings, None),
            "humanized",
        )

        explicit_fast = SimpleNamespace(policy="fast", fast=False)
        self.assertEqual(
            resolve_policy_name(explicit_fast, self.settings, None),
            "fast",
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
        self.assertEqual(
            [(x, y) for _, x, y in self.input.moves],
            [(100, 100), (200, 200)],
        )
        self.assertEqual(self.input.downs, [0.0, 0.0])

    def test_fast_policy_logs_frame_capture_timestamp(self):
        policy = FastExecutionPolicy(clock=self.clock)
        action = balloon_action(
            TargetType.NORMAL_BALLOON,
            BalloonColor.BLUE,
            (100, 100),
            50,
        )
        action["metadata"]["first_seen_timestamp"] = 123.456
        decision = self.engine.decide([action])

        self.clock.value = 200.0
        result = policy.tick(decision, self._context(), self.input)

        self.assertEqual(result.clicks_sent, 1)
        self.assertEqual(len(result.outcomes), 1)
        self.assertEqual(result.outcomes[0].target_first_seen_ts, 123.456)
        self.assertEqual(result.outcomes[0].actual_click_down_ts, 200.0)

    def test_executor_uses_injected_humanized_policy_without_fallback(self):
        executor = BalloonExecutor(
            FakeBehavior(),
            clock=self.clock,
            execution_policy=HumanizedExecutionPolicy(),
        )
        decision = self.engine.decide([
            balloon_action(
                TargetType.NORMAL_BALLOON,
                BalloonColor.BLUE,
                (100, 100),
                61,
            )
        ])

        with self.assertRaisesRegex(
            NotImplementedError,
            "HumanizedExecutionPolicy is not implemented until Phase B",
        ):
            executor.tick(decision.actions, None, self.input, 0, 0, True)

        self.assertEqual(self.input.downs, [])

    def test_humanized_policy_fails_loudly_without_fallback(self):
        policy = HumanizedExecutionPolicy()
        decision = self.engine.decide([
            balloon_action(
                TargetType.NORMAL_BALLOON,
                BalloonColor.BLUE,
                (100, 100),
                60,
            )
        ])

        with self.assertRaisesRegex(
            NotImplementedError,
            "HumanizedExecutionPolicy is not implemented until Phase B",
        ):
            policy.tick(decision, self._context(), self.input)

        self.assertEqual(self.input.downs, [])
        self.assertEqual(self.input.ups, [])


if __name__ == "__main__":
    unittest.main()
