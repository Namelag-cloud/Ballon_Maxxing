from __future__ import annotations

import unittest

from behavior.balloon_executor import BalloonExecutor
from behavior.model import BehavioralAction
from config.settings import Settings
from decision.mechanics import BalloonDecisionEngine
from game_profiles.balloon_types import (
    BalloonColor,
    TargetType,
    TwinAction,
)
from tracking.tracker import Tracker
from vision.detector import Detection
from vision.outline_detector import OutlineResult


class FakeClock:
    def __init__(self, value: float = 0.0) -> None:
        self.value = value

    def __call__(self) -> float:
        return self.value


class FakeInput:
    def __init__(self, clock: FakeClock, pair_delta_s: float = 0.005) -> None:
        self.clock = clock
        self.pair_delta_s = pair_delta_s
        self.moves: list[tuple[float, int, int]] = []
        self.downs: list[float] = []
        self.ups: list[float] = []
        self.pairs: list[tuple[tuple[int, int], tuple[int, int]]] = []

    def move_to(self, x: int, y: int) -> float:
        self.moves.append((self.clock(), x, y))
        return self.clock()

    def left_down(self) -> float:
        ts = self.clock()
        self.downs.append(ts)
        return ts

    def left_up(self) -> float:
        ts = self.clock()
        self.ups.append(ts)
        return ts

    def click_pair(
        self,
        first: tuple[int, int],
        second: tuple[int, int],
    ) -> tuple[float, float, float, float]:
        first_down = self.clock()
        second_down = first_down + self.pair_delta_s
        self.pairs.append((first, second))
        return first_down, first_down, second_down, second_down


class FakeRng:
    def normal(self, *_args) -> float:
        return 0.0


class FakeBehavior:
    rng = FakeRng()

    def __init__(self) -> None:
        self.plans = 0
        self.click_duration_s = 0.0

    def should_fail_super_g(self) -> bool:
        # This must no longer affect deterministic G precursor execution.
        return True

    def plan_action(self, **_kwargs) -> BehavioralAction:
        self.plans += 1
        return BehavioralAction(
            reaction_time_s=0.0,
            movement_time_s=0.0,
            click_duration_s=self.click_duration_s,
            trajectory=[],
            total_time_s=0.0,
            start_position=(0.0, 0.0),
            target_position=(0.0, 0.0),
            click_position=(0.0, 0.0),
            distance=0.0,
            target_width=20.0,
            fitts_id=0.0,
            endpoint_error_px=0.0,
            pause_duration_s=0.0,
            is_miss=False,
            is_delayed=False,
            session_modifier=1.0,
            warmup_factor=1.0,
            fatigue_factor=1.0,
        )


def balloon_action(
    target_type: TargetType,
    color: BalloonColor | None,
    center: tuple[int, int],
    target_id: int,
) -> dict:
    return {
        "target_type": target_type.value,
        "color_name": color.value if color else None,
        "center": center,
        "bbox": (center[0] - 10, center[1] - 10, 20, 20),
        "area": 400,
        "color_confidence": 0.95,
        "confidence": 0.95,
        "width": 20.0,
        "height": 20.0,
        "target_id": target_id,
        "metadata": {
            "semantic_confirmed": target_type in {
                TargetType.SPECIAL_SEQUENCE_BALLOON,
                TargetType.TWIN_BALLOON,
                TargetType.MULTICOLOR_SPECIAL,
            },
            "semantic_pair_id": (
                "integration-pair" if target_type == TargetType.TWIN_BALLOON else None
            ),
        },
    }


class TestBalloonExecutionIntegration(unittest.TestCase):
    def setUp(self) -> None:
        self.settings = Settings()
        self.settings.grouping.max_pair_distance = 200
        self.engine = BalloonDecisionEngine(self.settings)
        self.clock = FakeClock()
        self.behavior = FakeBehavior()
        self.input = FakeInput(self.clock)
        self.executor = BalloonExecutor(
            self.behavior,
            clock=self.clock,
            initial_mouse_position=(0, 0),
        )

    def _run_until_primary_complete(self, result_actions, g_action=None):
        result = None
        for _ in range(5):
            result = self.executor.tick(
                result_actions,
                g_action,
                self.input,
                0,
                0,
                True,
            )
            if result.completed_primary_action is not None or result.twin_completion is not None:
                return result
        self.fail("primary action did not complete")

    def test_fast_execution_dispatches_all_visible_normal_targets(self):
        executor = BalloonExecutor(
            self.behavior,
            clock=self.clock,
            initial_mouse_position=(0, 0),
            fast_execution=True,
        )
        decision = self.engine.decide([
            balloon_action(TargetType.NORMAL_BALLOON, BalloonColor.BLUE, (100, 100), 50),
            balloon_action(TargetType.NORMAL_BALLOON, BalloonColor.BLUE, (200, 200), 51),
            balloon_action(TargetType.NORMAL_BALLOON, BalloonColor.BLUE, (300, 300), 52),
        ])
        result = executor.tick(decision.actions, None, self.input, 0, 0, True)
        self.assertEqual(result.clicks_sent, 3)
        # Fast policy applies the configured humanization offset to normal clicks.
        # Verify the click remains within the current +/-2 px endpoint range.
        expected = [(100, 100), (200, 200), (300, 300)]
        actual = [(x, y) for _, x, y in self.input.moves]
        self.assertEqual(len(actual), len(expected))
        for (x, y), (expected_x, expected_y) in zip(actual, expected):
            self.assertLessEqual(abs(x - expected_x), 2)
            self.assertLessEqual(abs(y - expected_y), 2)
        self.assertEqual(len(self.input.downs), 3)

    def test_fast_execution_suppresses_recently_clicked_normal_target(self):
        executor = BalloonExecutor(
            self.behavior,
            clock=self.clock,
            initial_mouse_position=(0, 0),
            fast_execution=True,
        )
        first = self.engine.decide([
            balloon_action(TargetType.NORMAL_BALLOON, BalloonColor.BLUE, (100, 100), 60),
        ])
        result = executor.tick(first.actions, None, self.input, 0, 0, True)
        self.assertEqual(result.clicks_sent, 1)

        # The popped balloon is still visible in the next captured frame.
        # The tracker may keep the same ID or briefly emit a nearby replacement
        # ID, so both forms must be suppressed during the short render-lag window.
        same_id = self.engine.decide([
            balloon_action(TargetType.NORMAL_BALLOON, BalloonColor.BLUE, (100, 100), 60),
        ])
        result = executor.tick(same_id.actions, None, self.input, 0, 0, True)
        self.assertEqual(result.clicks_sent, 0)

        churned_id = self.engine.decide([
            balloon_action(TargetType.NORMAL_BALLOON, BalloonColor.BLUE, (108, 106), 61),
        ])
        result = executor.tick(churned_id.actions, None, self.input, 0, 0, True)
        self.assertEqual(result.clicks_sent, 0)
        self.assertEqual(len(self.input.downs), 1)

        # A genuinely different target outside the suppression radius is still clickable.
        different = self.engine.decide([
            balloon_action(TargetType.NORMAL_BALLOON, BalloonColor.BLUE, (140, 100), 62),
        ])
        result = executor.tick(different.actions, None, self.input, 0, 0, True)
        self.assertEqual(result.clicks_sent, 1)
        self.assertEqual(len(self.input.downs), 2)

        # Once the render-lag window expires, the same target may be clicked again.
        self.clock.value = 0.201
        visible_again = self.engine.decide([
            balloon_action(TargetType.NORMAL_BALLOON, BalloonColor.BLUE, (100, 100), 63),
        ])
        result = executor.tick(visible_again.actions, None, self.input, 0, 0, True)
        self.assertEqual(result.clicks_sent, 1)
        self.assertEqual(len(self.input.downs), 3)

    def test_normal_target_runs_behavior_scheduler_execution_path(self):
        decision = self.engine.decide([
            balloon_action(TargetType.NORMAL_BALLOON, BalloonColor.RED, (100, 100), 1)
        ])
        result = self._run_until_primary_complete(decision.actions)

        self.assertIsNotNone(result.completed_primary_action)
        self.assertEqual(self.input.downs, [0.0])
        self.assertEqual(self.input.ups, [0.0])

    def test_special_sequence_completes_only_after_scheduler_completion(self):
        self.engine.decide([
            balloon_action(TargetType.SPECIAL_SEQUENCE_BALLOON, BalloonColor.BLUE, (100, 100), 10),
            balloon_action(TargetType.SPECIAL_SEQUENCE_BALLOON, BalloonColor.RED, (900, 100), 11),
            balloon_action(TargetType.SPECIAL_SEQUENCE_BALLOON, BalloonColor.YELLOW, (1500, 100), 12),
            balloon_action(TargetType.SPECIAL_SEQUENCE_BALLOON, BalloonColor.GREEN, (300, 900), 13),
        ])
        decision = self.engine.decide([
            balloon_action(
                TargetType.SPECIAL_SEQUENCE_BALLOON,
                BalloonColor.BLUE,
                (100, 100),
                10,
            )
        ])
        result = self._run_until_primary_complete(decision.actions)
        self.assertEqual(result.completed_primary_action.target_id, 10)
        self.assertTrue(self.engine.complete_action(result.completed_primary_action))
        self.assertEqual(
            self.engine.state.required_special_color,
            BalloonColor.RED,
        )
        self.assertGreaterEqual(self.executor._idle_until, 0.100)

        next_decision = self.engine.decide([
            balloon_action(
                TargetType.SPECIAL_SEQUENCE_BALLOON,
                BalloonColor.RED,
                (120, 100),
                11,
            )
        ])
        self.assertEqual(next_decision.actions[0].target_id, 11)

    def test_twin_action_gets_two_behavior_plans_and_synchronized_down_events(self):
        decision = self.engine.decide([
            balloon_action(TargetType.TWIN_BALLOON, BalloonColor.YELLOW, (100, 100), 20),
            balloon_action(TargetType.TWIN_BALLOON, BalloonColor.YELLOW, (120, 100), 21),
        ])
        twin = decision.actions[0]
        self.assertIsInstance(twin, TwinAction)

        result = self._run_until_primary_complete(decision.actions)
        self.assertIsNotNone(result.twin_completion)
        self.assertEqual(self.behavior.plans, 2)
        self.assertEqual(len(self.input.pairs), 1)
        self.assertLessEqual(result.twin_completion.click_down_delta_ms, 30.0)
        self.assertEqual(
            {target_id for pair in self.input.pairs for target_id in pair},
            {(100, 100), (120, 100)},
        )

    def test_twin_scheduler_rejects_sync_violation(self):
        self.input.pair_delta_s = 0.031
        decision = self.engine.decide([
            balloon_action(TargetType.TWIN_BALLOON, BalloonColor.GREEN, (100, 100), 22),
            balloon_action(TargetType.TWIN_BALLOON, BalloonColor.GREEN, (120, 100), 23),
        ])
        with self.assertRaises(RuntimeError):
            self._run_until_primary_complete(decision.actions)

    def test_g_repeats_while_visible_and_refreshes_position(self):
        normal = self.engine.decide([
            balloon_action(TargetType.NORMAL_BALLOON, BalloonColor.BLUE, (300, 300), 30)
        ])
        g1 = self.engine.decide([
            balloon_action(TargetType.G_SYMBOL, None, (50, 50), 40),
            balloon_action(TargetType.NORMAL_BALLOON, BalloonColor.BLUE, (300, 300), 30),
        ]).g_action
        self.assertIsNotNone(g1)

        # Keep the first G action before its click phase so the position
        # refresh exercises the contract's cancellation path rather than
        # allowing an already-dispatched click to finish.
        self.executor.tick(normal.actions, g1, self.input, 0, 0, True)

        g2 = self.engine.decide([
            balloon_action(TargetType.G_SYMBOL, None, (80, 90), 41),
        ]).g_action
        self.assertIsNotNone(g2)
        result = self.executor.tick([], g2, self.input, 0, 0, True)

        self.assertFalse(result.g_click_completed)
        self.assertEqual(self.executor.latest_g.position, (80, 90))

        completed = False
        for _ in range(5):
            result = self.executor.tick([], g2, self.input, 0, 0, True)
            completed = completed or result.g_click_completed
            if completed:
                break
        self.assertTrue(completed)
        self.assertEqual(self.input.moves[-1][1:], (80, 90))

        # A still-pending G action is cancelled when vision reports that G
        # has disappeared; no counter is involved in this lifetime decision.
        self.executor.tick([], g2, self.input, 0, 0, True)
        disappeared = self.executor.tick([], None, self.input, 0, 0, True)
        self.assertFalse(disappeared.g_click_completed)

    def test_g_precursor_survives_tracker_id_churn_when_same_color_is_nearby(self):
        self.engine.decide([
            balloon_action(TargetType.SPECIAL_SEQUENCE_BALLOON, BalloonColor.BLUE, (100, 100), 1),
            balloon_action(TargetType.SPECIAL_SEQUENCE_BALLOON, BalloonColor.RED, (500, 500), 2),
            balloon_action(TargetType.SPECIAL_SEQUENCE_BALLOON, BalloonColor.YELLOW, (800, 100), 3),
            balloon_action(TargetType.SPECIAL_SEQUENCE_BALLOON, BalloonColor.GREEN, (1200, 700), 4),
        ])
        active = self.engine.decide([
            balloon_action(
                TargetType.SPECIAL_SEQUENCE_BALLOON,
                BalloonColor.BLUE,
                (100, 100),
                10,
            )
        ]).actions[0]
        self.assertTrue(self.engine.complete_action(active))
        red = self.engine.decide([
            balloon_action(
                TargetType.SPECIAL_SEQUENCE_BALLOON,
                BalloonColor.RED,
                (500, 500),
                101,
            )
        ]).actions[0]
        replacement = self.engine.decide([
            balloon_action(
                TargetType.SPECIAL_SEQUENCE_BALLOON,
                BalloonColor.RED,
                (530, 515),
                202,
            )
        ]).actions[0]
        self.assertTrue(red.metadata.get("g_precursor"))
        self.assertTrue(
            self.executor._special_precursor_replacement_observed(
                red,
                [replacement],
            )
        )

    def test_primary_action_is_cancelled_if_target_disappears_before_dispatch(self):
        action = self.engine.decide([
            balloon_action(TargetType.NORMAL_BALLOON, BalloonColor.RED, (100, 100), 31)
        ]).actions

        # Start the behavioral plan while the target is visible.
        self.executor.tick(action, None, self.input, 0, 0, True)
        self.assertFalse(self.executor.primary_scheduler.is_idle)

        # The next observation no longer contains that target. It must not
        # progress to mouse-down from the stale plan.
        self.clock.value = 1.0
        result = self.executor.tick([], None, self.input, 0, 0, True)

        self.assertFalse(result.completed_primary_action)
        self.assertTrue(self.executor.primary_scheduler.is_idle)
        self.assertEqual(self.input.downs, [])

    def test_shutdown_releases_held_button_and_marks_action_interrupted(self):
        self.behavior.click_duration_s = 1.0
        action = self.engine.decide([
            balloon_action(TargetType.NORMAL_BALLOON, BalloonColor.RED, (100, 100), 32)
        ]).actions

        self.executor.tick(action, None, self.input, 0, 0, True)
        self.clock.value = 0.1
        self.executor.tick(action, None, self.input, 0, 0, True)
        self.executor.tick(action, None, self.input, 0, 0, True)
        self.assertEqual(self.input.downs, [0.1])
        self.assertEqual(self.input.ups, [])

        outcomes = self.executor.shutdown(self.input)

        self.assertEqual(self.input.ups, [0.1])
        self.assertTrue(self.executor.primary_scheduler.is_idle)
        self.assertEqual(len(outcomes), 1)
        self.assertEqual(outcomes[0].execution_status, "interrupted")

    def test_tracker_preserves_game_metadata_without_interpreting_it(self):
        tracker = Tracker(self.settings)
        detection = Detection(
            bbox=(10, 10, 20, 20),
            center=(20, 20),
            area=400,
            color_name="unknown",
            color_confidence=1.0,
            outline=OutlineResult(False, 0.0, 0.0),
            tracking_class=TargetType.G_SYMBOL.value,
            metadata={"target_type": TargetType.G_SYMBOL.value, "marker": "opaque"},
        )
        tracks = tracker.update([detection])

        track = next(iter(tracks.values()))
        self.assertEqual(track.tracking_class, TargetType.G_SYMBOL.value)
        self.assertEqual(track.metadata["marker"], "opaque")


if __name__ == "__main__":
    unittest.main()
