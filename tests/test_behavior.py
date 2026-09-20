import sys
import os
import math
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from behavior.config import (
    BehaviorConfig, ReactionTimeConfig, MovementConfig,
    TrajectoryConfig, JitterConfig, ClickConfig, ErrorConfig,
    WarmupConfig, FatigueConfig, MicroPauseConfig, EndpointErrorConfig,
)
from behavior.distributions import SeededRNG, sample_lognormal_bounded, sample_fitts_mt
from behavior.movement import (
    TrajectoryPoint, MovementPlan, compute_fitts_id,
    generate_trajectory, apply_jitter, compute_endpoint_error,
)
from behavior.scheduler import SessionScheduler, SessionState
from behavior.model import BehavioralModel, BehavioralAction
from behavior.action_scheduler import ActionScheduler, ActionPhase


# ── Distribution Tests ──────────────────────────────────────────────

class TestSeededRNG:
    def test_reproducible_with_same_seed(self):
        rng1 = SeededRNG(42)
        rng2 = SeededRNG(42)
        vals1 = [rng1.random() for _ in range(100)]
        vals2 = [rng2.random() for _ in range(100)]
        assert vals1 == vals2

    def test_different_seeds_differ(self):
        rng1 = SeededRNG(42)
        rng2 = SeededRNG(99)
        vals1 = [rng1.random() for _ in range(50)]
        vals2 = [rng2.random() for _ in range(50)]
        assert vals1 != vals2

    def test_lognormal_positive(self):
        rng = SeededRNG(1)
        samples = [rng.lognormal(0, 0.5) for _ in range(1000)]
        assert all(s > 0 for s in samples)

    def test_normal_returns_float(self):
        rng = SeededRNG(1)
        val = rng.normal(0, 1)
        assert isinstance(val, float)

    def test_uniform_in_range(self):
        rng = SeededRNG(1)
        samples = [rng.uniform(2.0, 5.0) for _ in range(500)]
        assert all(2.0 <= s <= 5.0 for s in samples)


class TestLognormalBounded:
    def test_samples_are_positive(self):
        rng = SeededRNG(42)
        for _ in range(500):
            val = sample_lognormal_bounded(rng, 300, 0.25, 150, 1500)
            assert val > 0

    def test_minimum_respected(self):
        rng = SeededRNG(42)
        for _ in range(500):
            val = sample_lognormal_bounded(rng, 300, 0.25, 200, 1500)
            assert val >= 200

    def test_maximum_respected(self):
        rng = SeededRNG(42)
        for _ in range(500):
            val = sample_lognormal_bounded(rng, 300, 0.25, 150, 500)
            assert val <= 500

    def test_reproducible(self):
        rng1 = SeededRNG(42)
        rng2 = SeededRNG(42)
        vals1 = [sample_lognormal_bounded(rng1, 300, 0.25, 150, 1500) for _ in range(50)]
        vals2 = [sample_lognormal_bounded(rng2, 300, 0.25, 150, 1500) for _ in range(50)]
        assert vals1 == vals2

    def test_raises_on_invalid_params(self):
        rng = SeededRNG(1)
        try:
            sample_lognormal_bounded(rng, -1, 0.25, 0, 1000)
            assert False, "Should raise ValueError"
        except ValueError:
            pass

    def test_median_roughly_centered(self):
        rng = SeededRNG(42)
        samples = [sample_lognormal_bounded(rng, 300, 0.25, 10, 5000) for _ in range(2000)]
        median = sorted(samples)[len(samples) // 2]
        assert 200 < median < 500, f"Median {median} not near expected 300"


class TestFittsMovement:
    def test_larger_distance_increases_mt(self):
        rng = SeededRNG(42)
        mt_short = sample_fitts_mt(rng, 100, 50, 80, 70, 0.0)
        rng2 = SeededRNG(42)
        mt_long = sample_fitts_mt(rng2, 500, 50, 80, 70, 0.0)
        assert mt_long > mt_short

    def test_smaller_target_increases_mt(self):
        rng = SeededRNG(42)
        mt_large = sample_fitts_mt(rng, 300, 100, 80, 70, 0.0)
        rng2 = SeededRNG(42)
        mt_small = sample_fitts_mt(rng2, 300, 20, 80, 70, 0.0)
        assert mt_small > mt_large

    def test_zero_distance_returns_a(self):
        rng = SeededRNG(42)
        mt = sample_fitts_mt(rng, 0, 50, 80, 70, 0.0)
        assert mt == 80

    def test_variance_adds_stochasticity(self):
        rng = SeededRNG(42)
        mt1 = sample_fitts_mt(rng, 300, 50, 80, 70, 0.15)
        rng2 = SeededRNG(99)
        mt2 = sample_fitts_mt(rng2, 300, 50, 80, 70, 0.15)
        assert mt1 != mt2

    def test_no_variance_is_deterministic(self):
        mt1 = sample_fitts_mt(SeededRNG(42), 300, 50, 80, 70, 0.0)
        mt2 = sample_fitts_mt(SeededRNG(99), 300, 50, 80, 70, 0.0)
        assert mt1 == mt2


# ── Trajectory Tests ────────────────────────────────────────────────

class TestTrajectory:
    def test_starts_at_requested_position(self):
        rng = SeededRNG(42)
        pts = generate_trajectory(100, 100, 500, 500, 0.3, rng)
        assert len(pts) > 1
        assert abs(pts[0].x - 100) < 10
        assert abs(pts[0].y - 100) < 10

    def test_ends_near_endpoint(self):
        rng = SeededRNG(42)
        pts = generate_trajectory(100, 100, 500, 500, 0.3, rng)
        last = pts[-1]
        assert abs(last.x - 500) < 20
        assert abs(last.y - 500) < 20

    def test_produces_multiple_points(self):
        rng = SeededRNG(42)
        pts = generate_trajectory(0, 0, 100, 100, 0.2, rng, num_points=15)
        assert len(pts) >= 10

    def test_deterministic_with_same_seed(self):
        pts1 = generate_trajectory(0, 0, 400, 300, 0.3, SeededRNG(42))
        pts2 = generate_trajectory(0, 0, 400, 300, 0.3, SeededRNG(42))
        assert len(pts1) == len(pts2)
        for p1, p2 in zip(pts1, pts2):
            assert abs(p1.x - p2.x) < 0.001
            assert abs(p1.y - p2.y) < 0.001

    def test_different_seed_different_trajectory(self):
        pts1 = generate_trajectory(0, 0, 400, 300, 0.3, SeededRNG(42))
        pts2 = generate_trajectory(0, 0, 400, 300, 0.3, SeededRNG(99))
        assert not all(abs(p1.x - p2.x) < 0.1 for p1, p2 in zip(pts1, pts2))

    def test_no_nan_or_inf(self):
        rng = SeededRNG(42)
        pts = generate_trajectory(0, 0, 800, 600, 0.4, rng)
        for pt in pts:
            assert math.isfinite(pt.x)
            assert math.isfinite(pt.y)
            assert math.isfinite(pt.t)

    def test_zero_distance_returns_single_point(self):
        rng = SeededRNG(42)
        pts = generate_trajectory(100, 100, 100, 100, 0.3, rng)
        assert len(pts) <= 2

    def test_points_are_time_ordered(self):
        rng = SeededRNG(42)
        pts = generate_trajectory(0, 0, 500, 500, 0.3, rng)
        for i in range(1, len(pts)):
            assert pts[i].t >= pts[i - 1].t

    def test_final_timestamp_equals_requested_duration(self):
        rng = SeededRNG(42)
        for total_t in [0.1, 0.3, 0.5, 1.0, 2.0]:
            pts = generate_trajectory(0, 0, 500, 500, total_t, rng, num_points=20)
            assert len(pts) >= 2
            assert abs(pts[-1].t - total_t) < 0.001, (
                f"final {pts[-1].t} != requested {total_t}"
            )

    def test_segment_durations_sum_to_total(self):
        rng = SeededRNG(42)
        for total_t in [0.2, 0.5, 1.0]:
            pts = generate_trajectory(0, 0, 400, 300, total_t, rng, num_points=20)
            segment_sum = sum(pts[i].t - pts[i-1].t for i in range(1, len(pts)))
            assert abs(segment_sum - total_t) < 0.001, (
                f"segment sum {segment_sum} != total {total_t}"
            )

    def test_lateral_deviation_bounded(self):
        rng = SeededRNG(42)
        for _ in range(100):
            pts = generate_trajectory(
                0, 0, 500, 0, 0.5, rng,
                num_points=20, lateral_deviation_px=5.0,
            )
            for pt in pts:
                assert abs(pt.y) < 30, f"lateral deviation too large: {pt.y}"

    def test_velocity_finite(self):
        rng = SeededRNG(42)
        pts = generate_trajectory(0, 0, 500, 500, 0.3, rng, num_points=20)
        for pt in pts:
            assert math.isfinite(pt.velocity)
            assert pt.velocity >= 0

    def test_acceleration_deceleration_profile(self):
        rng = SeededRNG(42)
        pts = generate_trajectory(0, 0, 500, 0, 0.5, rng, num_points=20)
        mid = len(pts) // 2
        early_v = pts[3].velocity
        mid_v = pts[mid].velocity
        late_v = pts[-3].velocity
        assert mid_v > early_v * 0.5, "should accelerate then decelerate"
        assert mid_v > late_v * 0.5, "should accelerate then decelerate"

    def test_zero_total_time_returns_single_point(self):
        rng = SeededRNG(42)
        pts = generate_trajectory(0, 0, 100, 100, 0.0, rng)
        assert len(pts) == 1
        assert pts[0].t == 0.0

    def test_no_variance_produces_uniform_timing(self):
        rng = SeededRNG(42)
        pts = generate_trajectory(
            0, 0, 500, 0, 0.5, rng,
            num_points=11, velocity_variance=0.0,
        )
        for i in range(2, len(pts)):
            dt = pts[i].t - pts[i-1].t
            expected_dt = 0.5 / 10
            assert abs(dt - expected_dt) < 0.0001


# ── Jitter Tests ────────────────────────────────────────────────────

class TestJitter:
    def test_disabled_returns_same_position(self):
        rng = SeededRNG(42)
        x, y = apply_jitter(100, 200, rng, amplitude_px=0)
        assert x == 100 and y == 200

    def test_enabled_shifts_position(self):
        rng = SeededRNG(42)
        x, y = apply_jitter(100, 200, rng, amplitude_px=2.0)
        assert x != 100 or y != 200

    def test_shift_is_small(self):
        rng = SeededRNG(42)
        shifts = [apply_jitter(100, 200, rng, amplitude_px=1.5) for _ in range(100)]
        for x, y in shifts:
            assert abs(x - 100) < 10
            assert abs(y - 200) < 10


# ── Endpoint Error Tests ───────────────────────────────────────────

class TestEndpointError:
    def test_zero_error_returns_target(self):
        rng = SeededRNG(42)
        x, y = compute_endpoint_error(500, 500, 50, rng, 0.0, 0.0)
        assert x == 500 and y == 500

    def test_radial_error_non_negative(self):
        rng = SeededRNG(42)
        for _ in range(500):
            x, y = compute_endpoint_error(500, 500, 50, rng, 5.0, 12.0)
            r = math.sqrt((x - 500) ** 2 + (y - 500) ** 2)
            assert r >= 0

    def test_max_radius_respected(self):
        rng = SeededRNG(42)
        for _ in range(500):
            x, y = compute_endpoint_error(500, 500, 50, rng, 5.0, 12.0)
            r = math.sqrt((x - 500) ** 2 + (y - 500) ** 2)
            assert r <= 12.0 + 0.001

    def test_reproducible_with_same_seed(self):
        rng1 = SeededRNG(42)
        rng2 = SeededRNG(42)
        results1 = [compute_endpoint_error(500, 500, 50, rng1, 5.0, 12.0) for _ in range(50)]
        results2 = [compute_endpoint_error(500, 500, 50, rng2, 5.0, 12.0) for _ in range(50)]
        for (x1, y1), (x2, y2) in zip(results1, results2):
            assert abs(x1 - x2) < 0.001
            assert abs(y1 - y2) < 0.001

    def test_different_seeds_differ(self):
        rng1 = SeededRNG(42)
        rng2 = SeededRNG(99)
        results1 = [compute_endpoint_error(500, 500, 50, rng1, 5.0, 12.0) for _ in range(50)]
        results2 = [compute_endpoint_error(500, 500, 50, rng2, 5.0, 12.0) for _ in range(50)]
        any_diff = any(abs(x1 - x2) > 0.01 or abs(y1 - y2) > 0.01
                       for (x1, y1), (x2, y2) in zip(results1, results2))
        assert any_diff

    def test_approximately_zero_mean_dx_dy(self):
        rng = SeededRNG(42)
        n = 5000
        dx_sum = 0.0
        dy_sum = 0.0
        for _ in range(n):
            x, y = compute_endpoint_error(500, 500, 50, rng, 5.0, 12.0)
            dx_sum += x - 500
            dy_sum += y - 500
        mean_dx = dx_sum / n
        mean_dy = dy_sum / n
        assert abs(mean_dx) < 0.5, f"mean_dx={mean_dx}"
        assert abs(mean_dy) < 0.5, f"mean_dy={mean_dy}"

    def test_no_systematic_directional_bias(self):
        rng = SeededRNG(42)
        n = 5000
        quadrants = [0, 0, 0, 0]
        for _ in range(n):
            x, y = compute_endpoint_error(500, 500, 50, rng, 5.0, 12.0)
            dx, dy = x - 500, y - 500
            if dx >= 0 and dy >= 0:
                quadrants[0] += 1
            elif dx < 0 and dy >= 0:
                quadrants[1] += 1
            elif dx < 0 and dy < 0:
                quadrants[2] += 1
            else:
                quadrants[3] += 1
        expected = n / 4
        for q in quadrants:
            assert abs(q - expected) < expected * 0.15

    def test_disabled_returns_exact_target(self):
        rng = SeededRNG(42)
        for _ in range(100):
            x, y = compute_endpoint_error(500, 500, 50, rng, 0.0, 0.0)
            assert x == 500 and y == 500

    def test_scale_increases_spread(self):
        rng1 = SeededRNG(42)
        small_errors = [math.sqrt((compute_endpoint_error(500, 500, 50, rng1, 2.0, 12.0)[0] - 500)**2 +
                                  (compute_endpoint_error(500, 500, 50, rng1, 2.0, 12.0)[1] - 500)**2)
                        for _ in range(200)]
        rng2 = SeededRNG(42)
        large_errors = [math.sqrt((compute_endpoint_error(500, 500, 50, rng2, 8.0, 12.0)[0] - 500)**2 +
                                  (compute_endpoint_error(500, 500, 50, rng2, 8.0, 12.0)[1] - 500)**2)
                        for _ in range(200)]
        assert sum(large_errors) / len(large_errors) > sum(small_errors) / len(small_errors)


# ── Fitts ID Tests ──────────────────────────────────────────────────

class TestFittsId:
    def test_zero_distance(self):
        assert compute_fitts_id(0, 50) == 0.0

    def test_increases_with_distance(self):
        id1 = compute_fitts_id(100, 50)
        id2 = compute_fitts_id(500, 50)
        assert id2 > id1

    def test_decreases_with_width(self):
        id1 = compute_fitts_id(300, 20)
        id2 = compute_fitts_id(300, 100)
        assert id1 > id2


# ── Scheduler Tests ────────────────────────────────────────────────

class TestSessionScheduler:
    def _make_scheduler(self, warmup_enabled=True, fatigue_enabled=True):
        config = BehaviorConfig(
            warmup=WarmupConfig(enabled=warmup_enabled, duration_s=5.0, modifier=0.2),
            fatigue=FatigueConfig(enabled=fatigue_enabled, start_s=10.0, rate=0.1, maximum=0.5),
            micro_pauses=MicroPauseConfig(enabled=False),
        )
        rng = SeededRNG(42)
        return SessionScheduler(rng, config.warmup, config.fatigue, config.micro_pauses)

    def test_warmup_active_during_warmup(self):
        sched = self._make_scheduler()
        sched._session_start = time.perf_counter() - 1.0
        state = sched.update()
        assert state.warmup_factor > 0

    def test_warmup_zero_after_duration(self):
        sched = self._make_scheduler()
        sched._session_start = time.perf_counter() - 100.0
        state = sched.update()
        assert state.warmup_factor == 0.0

    def test_fatigue_zero_before_start(self):
        sched = self._make_scheduler()
        sched._session_start = time.perf_counter() - 1.0
        state = sched.update()
        assert state.fatigue_factor == 0.0

    def test_fatigue_increases_over_time(self):
        sched = self._make_scheduler()
        sched._session_start = time.perf_counter() - 15.0
        state = sched.update()
        assert state.fatigue_factor > 0

    def test_fatigue_capped(self):
        sched = self._make_scheduler()
        sched._session_start = time.perf_counter() - 999999.0
        state = sched.update()
        assert state.fatigue_factor <= 0.5

    def test_modifier_is_sum(self):
        sched = self._make_scheduler()
        sched._session_start = time.perf_counter() - 1.0
        state = sched.update()
        assert abs(state.modifier - (state.warmup_factor + state.fatigue_factor)) < 0.001

    def test_apply_modifier_increases_time(self):
        sched = self._make_scheduler()
        state = SessionState(modifier=0.2)
        result = sched.apply_modifier(100.0, state)
        assert result == 120.0

    def test_micro_pause_disabled(self):
        config = BehaviorConfig(
            micro_pauses=MicroPauseConfig(enabled=False),
        )
        rng = SeededRNG(42)
        sched = SessionScheduler(rng, config.warmup, config.fatigue, config.micro_pauses)
        assert sched.should_micro_pause() is None

    def test_micro_pause_stochastic(self):
        config = BehaviorConfig(
            micro_pauses=MicroPauseConfig(enabled=True, probability=1.0, minimum_ms=100, maximum_ms=200),
        )
        rng = SeededRNG(42)
        sched = SessionScheduler(rng, config.warmup, config.fatigue, config.micro_pauses)
        pause = sched.should_micro_pause()
        assert pause is not None
        assert 100 <= pause <= 200

    def test_should_miss_disabled(self):
        sched = self._make_scheduler()
        state = SessionState()
        assert not sched.should_miss(0.0, state)

    def test_should_miss_enabled(self):
        config = BehaviorConfig(
            errors=ErrorConfig(enabled=True, miss_probability=1.0),
        )
        rng = SeededRNG(42)
        sched = SessionScheduler(rng, config.warmup, config.fatigue, config.micro_pauses)
        state = SessionState()
        assert sched.should_miss(1.0, state)


# ── BehavioralModel Tests ──────────────────────────────────────────

class TestBehavioralModel:
    def _make_model(self, **overrides):
        defaults = dict(
            enabled=True,
            seed=42,
            reaction_time=ReactionTimeConfig(median_ms=300, sigma=0.25, minimum_ms=150, maximum_ms=1500),
            movement=MovementConfig(a_ms=80, b_ms=70, variance=0.0),
            trajectory=TrajectoryConfig(enabled=False),
            jitter=JitterConfig(enabled=False),
            click=ClickConfig(median_ms=80, sigma=0.25, minimum_ms=25, maximum_ms=250),
            errors=ErrorConfig(enabled=False),
            warmup=WarmupConfig(enabled=False),
            fatigue=FatigueConfig(enabled=False),
            micro_pauses=MicroPauseConfig(enabled=False),
        )
        defaults.update(overrides)
        config = BehaviorConfig(**defaults)
        return BehavioralModel(config)

    def test_plan_action_returns_behavioral_action(self):
        model = self._make_model()
        model.start_session()
        action = model.plan_action(100, 100, 500, 500, 50)
        assert isinstance(action, BehavioralAction)

    def test_reaction_time_positive(self):
        model = self._make_model()
        model.start_session()
        for _ in range(100):
            action = model.plan_action(0, 0, 500, 500, 50)
            assert action.reaction_time_s > 0

    def test_movement_time_positive(self):
        model = self._make_model()
        model.start_session()
        for _ in range(100):
            action = model.plan_action(0, 0, 500, 500, 50)
            assert action.movement_time_s > 0

    def test_click_duration_positive(self):
        model = self._make_model()
        model.start_session()
        for _ in range(100):
            action = model.plan_action(0, 0, 500, 500, 50)
            assert action.click_duration_s > 0

    def test_total_time_is_sum(self):
        model = self._make_model()
        model.start_session()
        action = model.plan_action(0, 0, 500, 500, 50)
        expected = action.reaction_time_s + action.movement_time_s + action.click_duration_s + action.pause_duration_s
        assert abs(action.total_time_s - expected) < 0.001

    def test_distance_computed(self):
        model = self._make_model()
        model.start_session()
        action = model.plan_action(100, 100, 400, 500, 50)
        expected_dist = math.sqrt(300**2 + 400**2)
        assert abs(action.distance - expected_dist) < 0.01

    def test_fitts_id_computed(self):
        model = self._make_model()
        model.start_session()
        action = model.plan_action(0, 0, 300, 0, 50)
        expected_id = math.log2(1 + 300 / 50)
        assert abs(action.fitts_id - expected_id) < 0.01

    def test_no_miss_when_disabled(self):
        model = self._make_model(errors=ErrorConfig(enabled=False))
        model.start_session()
        for _ in range(100):
            action = model.plan_action(0, 0, 500, 500, 50)
            assert not action.is_miss

    def test_non_miss_endpoint_error_is_bounded_relative_to_small_target(self):
        config = BehaviorConfig(
            enabled=True,
            seed=42,
            errors=ErrorConfig(enabled=True, miss_probability=0.0),
            endpoint_error=EndpointErrorConfig(
                enabled=True,
                scale_px=20.0,
                max_px=50.0,
                max_radius_fraction=0.5,
            ),
        )
        model = BehavioralModel(config)

        for _ in range(200):
            action = model.plan_action(0, 0, 100, 100, 20.0)
            error = math.hypot(
                action.click_position[0] - 100,
                action.click_position[1] - 100,
            )
            assert error <= 5.0 + 1e-9
            assert not action.is_miss

    def test_miss_possible_when_enabled(self):
        model = self._make_model(
            errors=ErrorConfig(enabled=True, miss_probability=0.5),
            seed=42,
        )
        model.start_session()
        misses = sum(
            1 for _ in range(200)
            if model.plan_action(0, 0, 500, 500, 50).is_miss
        )
        assert misses > 0

    def test_seed_reproduces_behavior_rng_but_not_session_profile(self):
        m1 = self._make_model(seed=42)
        m2 = self._make_model(seed=42)
        m1.start_session()
        m2.start_session()

        assert m1.behavioral_seed == m2.behavioral_seed == 42
        assert m1.profile_seed != m2.profile_seed
        assert m1.profile.as_dict() != m2.profile.as_dict()

    def test_different_seeds_differ(self):
        m1 = self._make_model(seed=42)
        m1.start_session()
        m2 = self._make_model(seed=99)
        m2.start_session()
        a1 = m1.plan_action(100, 100, 500, 500, 50)
        a2 = m2.plan_action(100, 100, 500, 500, 50)
        assert a1.reaction_time_s != a2.reaction_time_s

    def test_trajectory_generated_when_enabled(self):
        model = self._make_model(
            trajectory=TrajectoryConfig(enabled=True, points=10),
            jitter=JitterConfig(enabled=False),
        )
        model.start_session()
        action = model.plan_action(0, 0, 500, 500, 50)
        assert len(action.trajectory) > 0

    def test_no_trajectory_when_disabled(self):
        model = self._make_model(
            trajectory=TrajectoryConfig(enabled=False),
        )
        model.start_session()
        action = model.plan_action(0, 0, 500, 500, 50)
        assert len(action.trajectory) == 0

    def test_warmup_modifier_applied(self):
        model = self._make_model(
            warmup=WarmupConfig(enabled=True, duration_s=60.0, modifier=0.2),
            fatigue=FatigueConfig(enabled=False),
        )
        model.start_session()
        action = model.plan_action(0, 0, 500, 500, 50)
        assert action.session_modifier > 0

    def test_fatigue_factor_zero_early(self):
        model = self._make_model(
            warmup=WarmupConfig(enabled=False),
            fatigue=FatigueConfig(enabled=True, start_s=300.0),
        )
        model.start_session()
        action = model.plan_action(0, 0, 500, 500, 50)
        assert action.fatigue_factor == 0.0


# ── Config Tests ────────────────────────────────────────────────────

class TestBehaviorConfig:
    def test_defaults(self):
        config = BehaviorConfig()
        assert config.enabled is False
        assert config.seed == 12345
        assert config.reaction_time.median_ms == 300.0
        assert config.reaction_time.sigma == 0.45
        assert config.reaction_time.maximum_ms == 2500.0
        assert config.movement.a_ms == 80.0
        assert config.click.median_ms == 80.0
        assert config.endpoint_error.enabled is True
        assert config.endpoint_error.scale_px == 5.0
        assert config.endpoint_error.max_px == 12.0

    def test_from_dict(self):
        data = {
            "enabled": True,
            "seed": 999,
            "reaction_time": {"median_ms": 400, "sigma": 0.3},
            "movement": {"a_ms": 100, "b_ms": 80},
            "endpoint_error": {"enabled": False, "scale_px": 3.0, "max_px": 8.0},
        }
        config = BehaviorConfig.from_dict(data)
        assert config.enabled is True
        assert config.seed == 999
        assert config.reaction_time.median_ms == 400
        assert config.reaction_time.sigma == 0.3
        assert config.movement.a_ms == 100
        assert config.endpoint_error.enabled is False
        assert config.endpoint_error.scale_px == 3.0
        assert config.endpoint_error.max_px == 8.0

    def test_from_dict_empty(self):
        config = BehaviorConfig.from_dict({})
        assert config.enabled is False
        assert config.seed == 12345

    def test_from_dict_partial(self):
        data = {"enabled": True, "click": {"median_ms": 120}}
        config = BehaviorConfig.from_dict(data)
        assert config.enabled is True
        assert config.click.median_ms == 120
        assert config.reaction_time.median_ms == 300.0
        assert config.endpoint_error.scale_px == 5.0


class TestBenchmarkSmoke:
    def test_run_benchmark_small_sample(self):
        from benchmark_behavior import run_benchmark
        result = run_benchmark(n=100, seed=42)
        assert result["config"]["n_samples"] == 100
        assert result["config"]["seed"] == 42

        for key in ("reaction_time", "click_duration", "movement_time", "total_action_time", "endpoint_error"):
            s = result[key]
            assert s["n"] == 100
            assert s["mean"] > 0
            assert s["min"] <= s["p5"] <= s["p50"] <= s["p95"] <= s["max"]
            assert "skewness" in s

    def test_fitts_analysis_structure(self):
        from benchmark_behavior import run_benchmark
        result = run_benchmark(n=100, seed=42)
        fitts = result["fitts_analysis"]
        assert len(fitts["cells"]) == 15
        assert "monotonic_increasing_distance" in fitts
        assert "monotonic_decreasing_width" in fitts
        assert "predicted_vs_sampled_correlation" in fitts
        assert 0.9 < fitts["predicted_vs_sampled_correlation"] <= 1.0

    def test_deterministic_with_same_seed(self):
        from benchmark_behavior import run_benchmark
        r1 = run_benchmark(n=100, seed=42)
        r2 = run_benchmark(n=100, seed=42)
        assert r1["reaction_time"]["mean"] == r2["reaction_time"]["mean"]
        assert r1["movement_time"]["mean"] == r2["movement_time"]["mean"]

    def test_different_seeds_differ(self):
        from benchmark_behavior import run_benchmark
        r1 = run_benchmark(n=100, seed=42)
        r2 = run_benchmark(n=100, seed=99)
        assert r1["reaction_time"]["mean"] != r2["reaction_time"]["mean"]


class FakeClock:
    def __init__(self, start=0.0):
        self._time = start

    def __call__(self):
        return self._time

    def advance(self, dt):
        self._time += dt


class FakeInputBackend:
    def __init__(self):
        self.moves = []
        self.downs = 0
        self.ups = 0
        self._last_x = 960
        self._last_y = 540

    def move_to(self, x, y):
        self.moves.append((x, y))
        self._last_x = x
        self._last_y = y
        return self._time() if hasattr(self, '_time') else 0.0

    def left_down(self):
        self.downs += 1
        return 0.0

    def left_up(self):
        self.ups += 1
        return 0.0

    def click_screen(self, x, y):
        self.moves.append((x, y))
        self.downs += 1
        self.ups += 1
        return 0.0


class TestActionScheduler:
    def _make_action(self, reaction_s=0.3, movement_s=0.2, click_s=0.1, trajectory=None):
        from behavior.model import BehavioralAction
        return BehavioralAction(
            reaction_time_s=reaction_s,
            movement_time_s=movement_s,
            click_duration_s=click_s,
            trajectory=trajectory or [],
            total_time_s=reaction_s + movement_s + click_s,
            start_position=(0, 0),
            target_position=(500, 500),
            click_position=(500, 500),
            distance=500.0,
            target_width=50.0,
            fitts_id=3.0,
            endpoint_error_px=0.0,
            pause_duration_s=0.0,
            is_miss=False,
            is_delayed=False,
            session_modifier=0.0,
            warmup_factor=0.0,
            fatigue_factor=0.0,
        )

    def test_starts_idle(self):
        clock = FakeClock()
        scheduler = ActionScheduler(clock=clock)
        assert scheduler.is_idle
        assert scheduler.active_action is None

    def test_start_action_sets_active(self):
        clock = FakeClock()
        scheduler = ActionScheduler(clock=clock)
        action = self._make_action()
        scheduler.start_action(action, 500, 500)
        assert not scheduler.is_idle
        assert scheduler.active_action is not None

    def test_cannot_start_while_active(self):
        clock = FakeClock()
        scheduler = ActionScheduler(clock=clock)
        action1 = self._make_action(reaction_s=1.0)
        action2 = self._make_action(reaction_s=0.1)
        scheduler.start_action(action1, 500, 500)
        scheduler.start_action(action2, 300, 300)
        assert scheduler.active_action.target_x == 500

    def test_reaction_phase_lasting(self):
        clock = FakeClock()
        scheduler = ActionScheduler(clock=clock)
        input_backend = FakeInputBackend()
        action = self._make_action(reaction_s=0.5, movement_s=0.0, click_s=0.0)
        scheduler.start_action(action, 500, 500)

        clock.advance(0.1)
        scheduler.tick(input_backend)
        assert scheduler.active_action.phase == ActionPhase.WAITING_REACTION
        assert len(input_backend.moves) == 0

        clock.advance(0.3)
        scheduler.tick(input_backend)
        assert scheduler.active_action.phase == ActionPhase.WAITING_REACTION

    def test_transitions_to_moving_after_reaction(self):
        clock = FakeClock()
        scheduler = ActionScheduler(clock=clock)
        input_backend = FakeInputBackend()
        action = self._make_action(reaction_s=0.3, movement_s=0.0, click_s=0.0)
        scheduler.start_action(action, 500, 500)

        clock.advance(0.35)
        result = scheduler.tick(input_backend)
        assert result is None
        assert scheduler.active_action.phase == ActionPhase.MOVING

    def test_click_hold_duration(self):
        clock = FakeClock()
        scheduler = ActionScheduler(clock=clock)
        input_backend = FakeInputBackend()
        action = self._make_action(reaction_s=0.1, movement_s=0.0, click_s=0.2)
        scheduler.start_action(action, 500, 500)

        clock.advance(0.15)
        scheduler.tick(input_backend)
        assert scheduler.active_action.phase == ActionPhase.MOVING

        scheduler.tick(input_backend)
        assert scheduler.active_action.phase == ActionPhase.CLICK_DOWN

        scheduler.tick(input_backend)
        assert scheduler.active_action.phase == ActionPhase.CLICK_HOLD
        assert input_backend.downs == 1
        assert input_backend.ups == 0

        clock.advance(0.15)
        scheduler.tick(input_backend)
        assert scheduler.active_action.phase == ActionPhase.CLICK_HOLD

        clock.advance(0.15)
        result = scheduler.tick(input_backend)
        assert result is not None
        assert input_backend.ups == 1

    def test_action_reaches_complete(self):
        clock = FakeClock()
        scheduler = ActionScheduler(clock=clock)
        input_backend = FakeInputBackend()
        action = self._make_action(reaction_s=0.1, movement_s=0.0, click_s=0.1)
        scheduler.start_action(action, 500, 500)

        clock.advance(0.15)
        scheduler.tick(input_backend)

        scheduler.tick(input_backend)

        scheduler.tick(input_backend)

        clock.advance(0.2)
        result = scheduler.tick(input_backend)
        assert result is not None
        assert scheduler.is_idle

    def test_cancel_stops_active_action(self):
        clock = FakeClock()
        scheduler = ActionScheduler(clock=clock)
        input_backend = FakeInputBackend()
        action = self._make_action(reaction_s=1.0, movement_s=1.0, click_s=1.0)
        scheduler.start_action(action, 500, 500)

        clock.advance(0.5)
        result = scheduler.cancel()
        assert result is not None
        assert scheduler.is_idle

    def test_trajectory_executed_over_time(self):
        from behavior.movement import TrajectoryPoint
        trajectory = [
            TrajectoryPoint(x=100, y=100, t=0.0, velocity=100),
            TrajectoryPoint(x=200, y=200, t=0.1, velocity=100),
            TrajectoryPoint(x=500, y=500, t=0.2, velocity=0),
        ]
        clock = FakeClock()
        scheduler = ActionScheduler(clock=clock)
        input_backend = FakeInputBackend()
        action = self._make_action(reaction_s=0.1, movement_s=0.3, click_s=0.0, trajectory=trajectory)
        scheduler.start_action(action, 500, 500)

        clock.advance(0.15)
        scheduler.tick(input_backend)
        assert scheduler.active_action.phase == ActionPhase.MOVING

        scheduler.tick(input_backend)
        assert len(input_backend.moves) == 1
        assert input_backend.moves[0] == (100, 100)

        clock.advance(0.1)
        scheduler.tick(input_backend)
        assert len(input_backend.moves) == 2
        assert input_backend.moves[1] == (200, 200)

        clock.advance(0.1)
        scheduler.tick(input_backend)
        assert len(input_backend.moves) == 3

    def test_no_click_without_reaction_elapsed(self):
        clock = FakeClock()
        scheduler = ActionScheduler(clock=clock)
        input_backend = FakeInputBackend()
        action = self._make_action(reaction_s=0.5, movement_s=0.0, click_s=0.1)
        scheduler.start_action(action, 500, 500)

        clock.advance(0.1)
        scheduler.tick(input_backend)
        assert input_backend.downs == 0
        assert input_backend.ups == 0

    def test_disabled_behavior_uses_existing_path(self):
        from behavior.config import BehaviorConfig
        config = BehaviorConfig(enabled=False)
        model = BehavioralModel(config)
        assert not config.enabled
        scheduler = ActionScheduler()
        assert scheduler.is_idle
