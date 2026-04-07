import unittest

import numpy as np

from trajgen import HumanLikeProfile, generate_batch, generate_default_demo, generate_human_like_with_plan


class TrajgenTests(unittest.TestCase):
    def test_default_demo_milestones(self) -> None:
        tr = generate_default_demo(dt=0.05)
        q = np.array([0.0, 2.0, 6.0, 8.0], dtype=np.float64)
        pos, _, _, yaw = tr.sample_batch(q)

        self.assertTrue(np.allclose(pos[0], np.array([0.0, 0.0, -1.0]), atol=1e-6))
        self.assertTrue(np.allclose(pos[1], np.array([0.0, 0.0, -1.0]), atol=5e-2))
        self.assertTrue(np.allclose(pos[2], np.array([5.0, 0.0, -1.0]), atol=5e-2))
        self.assertTrue(np.allclose(pos[3], np.array([5.0, 0.0, -1.0]), atol=5e-2))
        self.assertTrue(np.allclose(yaw, np.zeros_like(yaw), atol=1e-6))

    def test_batch_reproducibility(self) -> None:
        seeds = np.array([1, 2, 3], dtype=np.int64)
        a = generate_batch(seeds=seeds, dt=0.05, duration=12.0)
        b = generate_batch(seeds=seeds, dt=0.05, duration=12.0)
        self.assertTrue(np.allclose(a.t, b.t))
        self.assertTrue(np.allclose(a.pos, b.pos))
        self.assertTrue(np.allclose(a.yaw, b.yaw))

    def test_batch_shapes(self) -> None:
        seeds = np.array([10, 11], dtype=np.int64)
        tr = generate_batch(seeds=seeds, dt=0.1, duration=5.0)
        self.assertEqual(tr.pos.shape[0], 2)
        self.assertEqual(tr.pos.shape[1], tr.t.shape[0])
        self.assertEqual(tr.pos.shape[2], 3)
        self.assertEqual(tr.yaw.shape, (2, tr.t.shape[0]))

    def test_human_like_segment_count_and_final_hover(self) -> None:
        _, segments = generate_human_like_with_plan(seed=7, dt=0.05, duration=25.0)
        move_count = sum(1 for s in segments if s.kind in ("smooth_move", "arc_cw", "arc_ccw"))
        self.assertIn(move_count, (5, 6))
        self.assertEqual(segments[-1].kind, "final_hover")

    def test_human_like_yaw_rate_limited(self) -> None:
        p = HumanLikeProfile(yaw_rate_limit=np.deg2rad(60.0))
        tr, _ = generate_human_like_with_plan(seed=11, dt=0.05, duration=25.0, profile=p)
        yaw_unwrapped = np.unwrap(tr.yaw)
        yaw_rate = np.gradient(yaw_unwrapped, tr.t, edge_order=2)
        self.assertLessEqual(float(np.max(np.abs(yaw_rate))), float(p.yaw_rate_limit + 1e-6))


if __name__ == "__main__":
    unittest.main()
