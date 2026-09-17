from __future__ import annotations

import unittest

from behavior.action_scheduler import ActionPhase, ActionScheduler
from behavior.model import BehavioralAction
from behavior.movement import TrajectoryPoint


class FakeClock:
    def __init__(self, value: float = 0.0) -> None:
        self.value = value

    def __call__(self) -> float:
        return self.value


class FakeInput:
    def __init__(self, clock: FakeClock) -> None:
        self.clock = clock
        self.moves: list[tuple[float, int, int]] = []
        self.downs: list[float] = []
        self.ups: list[float] = []

    def move_to(self, x: int, y: int) -> float:
        self.moves.append((self.clock(), x, y))
        return self.clock()

    def left_down(self) -> float:
        self.downs.append(self.clock())
        return self.clock()

    def left_up(self) -> float:
        self.ups.append(self.clock())
        return self.clock()


class TestActionScheduler(unittest.TestCase):
    def _action(self) -> BehavioralAction:
        trajectory = [
            TrajectoryPoint(0, 0, 0.00, 0),
            TrajectoryPoint(20, 0, 0.05, 0),
            TrajectoryPoint(40, 0, 0.10, 0),
            TrajectoryPoint(60, 0, 0.15, 0),
            TrajectoryPoint(80, 0, 0.20, 0),
        ]
        return BehavioralAction(
            reaction_time_s=0.0,
            movement_time_s=0.20,
            click_duration_s=0.05,
            trajectory=trajectory,
            total_time_s=0.25,
            start_position=(0, 0),
            target_position=(80, 0),
            click_position=(80, 0),
            distance=80.0,
            target_width=20.0,
            fitts_id=2.32,
            endpoint_error_px=0.0,
            pause_duration_s=0.0,
            is_miss=False,
            is_delayed=False,
            session_modifier=1.0,
            warmup_factor=1.0,
            fatigue_factor=1.0,
        )

    def test_overdue_points_are_consumed_in_one_tick(self):
        clock = FakeClock()
        input_backend = FakeInput(clock)
        scheduler = ActionScheduler(clock=clock)
        scheduler.start_action(self._action(), 80, 0)

        # First poll transitions reaction -> movement.
        scheduler.tick(input_backend)

        # Simulate a delayed 220 ms poll. All overdue points must be consumed
        # immediately instead of one point per scheduler tick.
        clock.value = 0.22
        scheduler.tick(input_backend)

        active = scheduler.active_action
        self.assertIsNotNone(active)
        assert active is not None
        self.assertEqual(active.phase, ActionPhase.CLICK_DOWN)
        self.assertEqual(active.trajectory_index, 5)
        self.assertAlmostEqual(active.timestamps.movement_end, 0.22, places=9)

        # Five trajectory points plus the final exact-target correction move.
        self.assertEqual(len(input_backend.moves), 6)

    def test_actual_movement_tracks_planned_time_with_polling_jitter(self):
        clock = FakeClock()
        input_backend = FakeInput(clock)
        scheduler = ActionScheduler(clock=clock)
        planned = self._action()
        scheduler.start_action(planned, 80, 0)
        scheduler.tick(input_backend)

        clock.value = 0.23
        scheduler.tick(input_backend)

        timestamps = scheduler.active_action.timestamps  # type: ignore[union-attr]
        actual = timestamps.movement_end - timestamps.movement_start
        self.assertAlmostEqual(actual, 0.23, places=9)
        self.assertAlmostEqual(planned.movement_time_s, 0.20, places=9)
        # One delayed poll may add wall-clock latency, but the scheduler must
        # not multiply that delay by the number of trajectory points.
        self.assertLess(actual - planned.movement_time_s, 0.05)


if __name__ == "__main__":
    unittest.main()
