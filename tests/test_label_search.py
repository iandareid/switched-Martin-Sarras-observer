import unittest

import numpy as np

from observer_labeling.data.dataset import TrajectoryDataset
from observer_labeling.data.jax_dataset import to_jax_dataset
from observer_labeling.eval.label_search import (
    build_mahony_params_from_config,
    build_search_problem_from_config,
    generate_decision_boundaries,
    label_trajectory,
    profile_solver,
    rollout_action,
    solve_decision,
    solve_decision_reference,
)


def _make_dataset(num_samples: int = 80) -> TrajectoryDataset:
    t = np.arange(num_samples, dtype=np.float64) / 400.0
    accel = np.tile(np.array([[0.0, 0.0, -1.0]], dtype=np.float64), (num_samples, 1))
    mag = np.tile(np.array([[1.0, 0.0, 0.0]], dtype=np.float64), (num_samples, 1))
    return TrajectoryDataset(
        t=t,
        cmd_rates=np.zeros((num_samples, 3), dtype=np.float64),
        throttle=np.zeros((num_samples, 1), dtype=np.float64),
        gyro=np.zeros((num_samples, 3), dtype=np.float64),
        accel=accel,
        mag_body=mag,
        true_quat=np.tile(np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float64), (num_samples, 1)),
        true_bias=np.zeros((num_samples, 3), dtype=np.float64),
        true_gyro_bias=np.tile(np.array([[0.1, 0.0, 0.0]], dtype=np.float64), (num_samples, 1)),
        true_accel_bias=np.tile(np.array([[0.0, 0.2, 0.0]], dtype=np.float64), (num_samples, 1)),
    )


def _make_config() -> dict:
    return {
        "env": {
            "estimator_rate_hz": 400.0,
            "decision_rate_hz": 20.0,
            "hold_steps": 5,
        },
        "label_search": {
            "w_attitude": 1.0,
            "w_gyro_bias": 1.0,
            "w_accel_bias": 1.0,
            "attitude_pre_break_gain": 1.0,
            "attitude_break_norm": 0.07,
            "attitude_transition_width": 0.01,
            "attitude_post_break_gain": 2.0,
            "gyro_bias_pre_break_gain": 1.0,
            "gyro_bias_break_norm": 0.04,
            "gyro_bias_transition_width": 0.01,
            "gyro_bias_post_break_gain": 2.0,
            "accel_bias_pre_break_gain": 1.0,
            "accel_bias_break_norm": 0.04,
            "accel_bias_transition_width": 0.01,
            "accel_bias_post_break_gain": 2.0,
            "mahony": {
                "k_p": 2.0,
                "k_i": 0.1,
                "accel_weight": 1.0,
                "mag_weight": 1.0,
                "accel_gate_margin_mps2": 0.5,
                "g_ref_mps2": 9.81,
            },
        },
    }


class LabelSearchTests(unittest.TestCase):
    def test_rollout_cost_increases_with_stronger_bias_gains(self) -> None:
        dataset = to_jax_dataset(_make_dataset())
        base_config = _make_config()
        tuned_config = _make_config()
        tuned_config["label_search"]["gyro_bias_post_break_gain"] = 8.0
        tuned_config["label_search"]["accel_bias_post_break_gain"] = 8.0
        base_problem, _ = build_search_problem_from_config(base_config, dataset)
        tuned_problem, _ = build_search_problem_from_config(tuned_config, dataset)
        boundaries = generate_decision_boundaries(base_problem)
        boundary = boundaries[0]

        base_rollout = rollout_action(base_problem, boundary.est_state, boundary.sample_idx, 0)
        tuned_rollout = rollout_action(tuned_problem, boundary.est_state, boundary.sample_idx, 0)

        self.assertGreater(tuned_rollout.interval_cost, base_rollout.interval_cost)

    def test_solve_decision_matches_reference_on_small_depth(self) -> None:
        dataset = to_jax_dataset(_make_dataset())
        problem, _ = build_search_problem_from_config(_make_config(), dataset)
        boundaries = generate_decision_boundaries(problem)
        boundary = boundaries[0]

        exact = solve_decision_reference(problem, boundary.est_state, boundary.sample_idx, depth=3)
        fast = solve_decision(problem, boundary.est_state, boundary.sample_idx, depth=3)

        np.testing.assert_allclose(fast.root_costs, exact.root_costs, rtol=1e-6, atol=1e-6)
        self.assertEqual(fast.best_action, exact.best_action)
        self.assertTrue(fast.completed)

    def test_generate_decision_boundaries_is_monotonic(self) -> None:
        dataset = to_jax_dataset(_make_dataset())
        problem, _ = build_search_problem_from_config(_make_config(), dataset)
        boundaries = generate_decision_boundaries(problem)

        self.assertGreater(len(boundaries), 1)
        self.assertEqual(boundaries[0].sample_idx, 0)
        self.assertTrue(all(a.sample_idx < b.sample_idx for a, b in zip(boundaries, boundaries[1:], strict=False)))

    def test_profile_solver_returns_projection(self) -> None:
        dataset = to_jax_dataset(_make_dataset())
        problem, _ = build_search_problem_from_config(_make_config(), dataset)
        boundaries = generate_decision_boundaries(problem)[:3]

        profiles, projection = profile_solver(problem, boundaries, depth=2)

        self.assertEqual(len(profiles), 3)
        self.assertEqual(projection.num_decisions, 3)
        self.assertTrue(np.isfinite(projection.mean_decision_sec))

    def test_label_trajectory_returns_consistent_trace(self) -> None:
        dataset = to_jax_dataset(_make_dataset())
        config = _make_config()
        problem, _ = build_search_problem_from_config(config, dataset)
        trace = label_trajectory(
            problem,
            depth=2,
            mahony_params=build_mahony_params_from_config(config, dataset),
        )

        self.assertEqual(trace.actions.shape[0], trace.decision_t.shape[0])
        self.assertEqual(trace.actions.shape[0], trace.root_costs.shape[0])
        self.assertEqual(trace.root_costs.shape[1], 3)
        self.assertGreater(trace.t.shape[0], trace.actions.shape[0])
        self.assertEqual(trace.search_completed.shape[0], trace.actions.shape[0])


if __name__ == "__main__":
    unittest.main()
