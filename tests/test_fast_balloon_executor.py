import unittest

from behavior.balloon_executor import BalloonExecutor
from game_profiles.balloon_types import BalloonAction, BalloonColor, TargetType


class RecordingInput:
    def __init__(self):
        self.events = []
        self._ts = 1.0

    def _timestamp(self):
        self._ts += 0.001
        return self._ts

    def move_to(self, x, y):
        self.events.append(("move", x, y))
        return self._timestamp()

    def left_down(self):
        self.events.append(("down",))
        return self._timestamp()

    def left_up(self):
        self.events.append(("up",))
        return self._timestamp()

    def click_pair(self, first, second, inter_click_delay_s=0.0):
        self.events.append(("pair", first, second, inter_click_delay_s))
        first_down = self.left_down()
        first_up = self.left_up()
        second_down = self.left_down()
        second_up = self.left_up()
        return first_down, first_up, second_down, second_up


def make_balloon(target_id=1, position=(100, 200)):
    return BalloonAction(
        target_id=target_id,
        target_type=TargetType.NORMAL_BALLOON,
        color=BalloonColor.RED,
        position=position,
        target_width=40.0,
    )


class FastBalloonExecutorTests(unittest.TestCase):
    def test_fast_path_dispatches_single_click_without_behavioral_waits(self):
        backend = RecordingInput()
        executor = BalloonExecutor(object(), fast_execution=True)

        result = executor.tick(
            (make_balloon(),),
            None,
            backend,
            0,
            0,
            True,
        )

        self.assertEqual(result.clicks_sent, 1)
        self.assertEqual(result.completed_primary_action.target_id, 1)
        self.assertEqual([event[0] for event in backend.events], ["move", "down", "up"])
        self.assertIsNotNone(result.action_timestamps)
        self.assertEqual(result.action_timestamps.click_down - result.action_timestamps.movement_start, 0.001)
        self.assertEqual(result.outcomes[0].execution_status, "fast")

    def test_fast_path_executes_twin_with_zero_inter_click_delay(self):
        backend = RecordingInput()
        first = make_balloon(1, (100, 200))
        second = make_balloon(2, (160, 200))
        from game_profiles.balloon_types import TwinAction

        executor = BalloonExecutor(object(), fast_execution=True)
        result = executor.tick(
            (TwinAction(targets=(first, second)),),
            None,
            backend,
            0,
            0,
            True,
        )

        self.assertEqual(result.clicks_sent, 2)
        self.assertEqual(backend.events[0], ("pair", (100, 200), (160, 200), 0.0))
        self.assertEqual(result.twin_completion.intended_click_down_delta_ms, 0.0)
        self.assertEqual(result.outcomes[0].execution_status, "fast")
        self.assertEqual(result.outcomes[1].execution_status, "fast")


if __name__ == "__main__":
    unittest.main()
