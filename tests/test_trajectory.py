import unittest

import numpy as np

from trajectory import TrajectoryBuffer, wrap_angle


class TrajectoryTests(unittest.TestCase):
    def test_interpolation_and_hold(self) -> None:
        t = np.array([0.0, 1.0, 2.0], dtype=np.float64)
        pos = np.array([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0], [2.0, 4.0, 6.0]], dtype=np.float64)
        yaw = np.array([0.0, np.pi / 2.0, np.pi], dtype=np.float64)
        traj = TrajectoryBuffer.from_samples(t, pos, yaw)

        q = np.array([0.5, 3.0], dtype=np.float64)
        pos_q, _, _, yaw_q = traj.sample_batch(q)
        self.assertTrue(np.allclose(pos_q[0], np.array([0.5, 1.0, 1.5])))
        self.assertTrue(np.allclose(pos_q[1], np.array([2.0, 4.0, 6.0])))
        self.assertAlmostEqual(yaw_q[0], np.pi / 4.0, places=6)
        self.assertAlmostEqual(yaw_q[1], -np.pi, places=6)

    def test_end_hold_zero_velocity_acceleration(self) -> None:
        t = np.array([0.0, 1.0, 2.0], dtype=np.float64)
        pos = np.array([[0.0, 0.0, -1.0], [1.0, 0.0, -1.0], [2.0, 0.0, -1.0]], dtype=np.float64)
        yaw = np.array([0.0, 0.2, 0.4], dtype=np.float64)
        traj = TrajectoryBuffer.from_samples(t, pos, yaw)

        q = np.array([2.0, 3.0], dtype=np.float64)
        pos_q, vel_q, acc_q, yaw_q = traj.sample_batch(q)
        self.assertTrue(np.allclose(pos_q[0], pos[-1], atol=1e-9))
        self.assertTrue(np.allclose(pos_q[1], pos[-1], atol=1e-9))
        self.assertTrue(np.allclose(vel_q[0], np.zeros(3), atol=1e-9))
        self.assertTrue(np.allclose(vel_q[1], np.zeros(3), atol=1e-9))
        self.assertTrue(np.allclose(acc_q[0], np.zeros(3), atol=1e-9))
        self.assertTrue(np.allclose(acc_q[1], np.zeros(3), atol=1e-9))
        self.assertAlmostEqual(float(yaw_q[1]), float(wrap_angle(np.array([yaw[-1]], dtype=np.float64))[0]), places=9)

    def test_wrap_angle(self) -> None:
        angles = np.array([-4 * np.pi, -np.pi, 0.0, np.pi, 5 * np.pi], dtype=np.float64)
        wrapped = wrap_angle(angles)
        self.assertTrue(np.all(wrapped <= np.pi))
        self.assertTrue(np.all(wrapped >= -np.pi))

    def test_demo_alias(self) -> None:
        traj = TrajectoryBuffer.demo(dt=0.05)
        p, _, _, _ = traj.sample_batch(np.array([0.0, 8.0], dtype=np.float64))
        self.assertTrue(np.allclose(p[0], np.array([0.0, 0.0, -1.0]), atol=1e-6))
        self.assertTrue(np.allclose(p[1], np.array([5.0, 0.0, -1.0]), atol=5e-2))


if __name__ == "__main__":
    unittest.main()
