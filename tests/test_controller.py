import unittest

import numpy as np

from controller import (
    CascadedControllerConfig,
    CascadedControllerStage,
    ControlMode,
    ControlStack,
    ControlStackConfig,
    ControlStackState,
    ControllerCommandBatch,
    ControllerState,
    EstimatedStateBatch,
    FollowerState,
    Mixer,
    TrajectoryCommandBatch,
    TrajectoryFollowerConfig,
    TrajectoryFollowerStage,
    wrap_angle,
)


def _quat_from_rpy(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr = np.cos(0.5 * roll)
    sr = np.sin(0.5 * roll)
    cp = np.cos(0.5 * pitch)
    sp = np.sin(0.5 * pitch)
    cy = np.cos(0.5 * yaw)
    sy = np.sin(0.5 * yaw)
    return np.array(
        [
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
        ],
        dtype=np.float64,
    )


class ControllerTests(unittest.TestCase):
    def test_wrap_angle(self) -> None:
        angles = np.array([-4 * np.pi, -np.pi, 0.0, np.pi, 5 * np.pi], dtype=np.float64)
        wrapped = wrap_angle(angles)
        self.assertTrue(np.all(wrapped <= np.pi))
        self.assertTrue(np.all(wrapped >= -np.pi))

    def test_mixer_hover_split(self) -> None:
        rotor_pos = np.array(
            [
                [-0.14, -0.18, 0.05],
                [-0.14, 0.18, 0.05],
                [0.14, 0.18, 0.08],
                [0.14, -0.18, 0.08],
            ],
            dtype=np.float64,
        )
        yaw_coeff = np.array([-0.0201, 0.0201, -0.0201, 0.0201], dtype=np.float64)
        mixer = Mixer.from_geometry(
            rotor_pos_body=rotor_pos,
            yaw_coeff=yaw_coeff,
            cmd_min=np.zeros(4, dtype=np.float64),
            cmd_max=np.full(4, 20.0, dtype=np.float64),
        )
        out = mixer.mix(thrust=np.array([8.0]), tau_body=np.array([[0.0, 0.0, 0.0]]))
        self.assertTrue(np.allclose(out.motor[0], np.array([2.0, 2.0, 2.0, 2.0]), atol=1e-6))
        self.assertFalse(bool(out.saturated[0]))

    def test_trajectory_follower_yaw_wrap(self) -> None:
        cfg = TrajectoryFollowerConfig()
        stage = TrajectoryFollowerStage(cfg)
        state = FollowerState.zeros(batch_size=1)

        est = EstimatedStateBatch(
            pos_world=np.zeros((1, 3), dtype=np.float64),
            vel_world=np.zeros((1, 3), dtype=np.float64),
            quat_world_body=_quat_from_rpy(0.0, 0.0, np.pi - 0.01)[None, :],
            omega_body=np.zeros((1, 3), dtype=np.float64),
        )
        traj = TrajectoryCommandBatch(
            pos_ref_world=np.zeros((1, 3), dtype=np.float64),
            vel_ref_world=np.zeros((1, 3), dtype=np.float64),
            acc_ref_world=np.zeros((1, 3), dtype=np.float64),
            yaw_ref=np.array([-np.pi + 0.01], dtype=np.float64),
            yaw_rate_ref=np.zeros((1,), dtype=np.float64),
        )

        cmd, _, _ = stage.step(state=state, traj=traj, est=est, dt=0.01, mass=1.0)
        self.assertEqual(int(cmd.mode[0]), int(ControlMode.ROLL_PITCH_YAWRATE_THRUST_TO_MIXER))
        self.assertLess(abs(float(cmd.cmd3[0])), 4.0 + 1e-9)

    def test_controller_stage_saturation(self) -> None:
        rotor_pos = np.array(
            [
                [-0.14, -0.18, 0.05],
                [-0.14, 0.18, 0.05],
                [0.14, 0.18, 0.08],
                [0.14, -0.18, 0.08],
            ],
            dtype=np.float64,
        )
        yaw_coeff = np.array([-0.0201, 0.0201, -0.0201, 0.0201], dtype=np.float64)
        mixer = Mixer.from_geometry(
            rotor_pos_body=rotor_pos,
            yaw_coeff=yaw_coeff,
            cmd_min=np.zeros(4, dtype=np.float64),
            cmd_max=np.full(4, 0.1, dtype=np.float64),
        )
        stage = CascadedControllerStage(CascadedControllerConfig(), mixer)
        state = ControllerState.zeros(batch_size=1)
        est = EstimatedStateBatch(
            pos_world=np.zeros((1, 3), dtype=np.float64),
            vel_world=np.zeros((1, 3), dtype=np.float64),
            quat_world_body=_quat_from_rpy(0.2, -0.2, 0.0)[None, :],
            omega_body=np.zeros((1, 3), dtype=np.float64),
        )
        cmd = ControllerCommandBatch(
            mode=np.array([int(ControlMode.ROLL_PITCH_YAWRATE_THRUST_TO_MIXER)], dtype=np.int64),
            cmd1=np.array([0.0], dtype=np.float64),
            cmd2=np.array([0.0], dtype=np.float64),
            cmd3=np.array([0.0], dtype=np.float64),
            cmd4=np.array([6.0], dtype=np.float64),
        )

        out, _, _ = stage.step(state=state, cmd=cmd, est=est, dt=0.0025)
        self.assertTrue(bool(out.saturated[0]))

    def test_control_stack_batch_shapes(self) -> None:
        rotor_pos = np.array(
            [
                [-0.14, -0.18, 0.05],
                [-0.14, 0.18, 0.05],
                [0.14, 0.18, 0.08],
                [0.14, -0.18, 0.08],
            ],
            dtype=np.float64,
        )
        yaw_coeff = np.array([-0.0201, 0.0201, -0.0201, 0.0201], dtype=np.float64)
        mixer = Mixer.from_geometry(
            rotor_pos_body=rotor_pos,
            yaw_coeff=yaw_coeff,
            cmd_min=np.zeros(4, dtype=np.float64),
            cmd_max=np.full(4, 20.0, dtype=np.float64),
        )

        cfg = ControlStackConfig()
        stack = ControlStack(cfg=cfg, mixer=mixer, mass=1.5)
        state = ControlStackState.zeros(batch_size=2)

        est = EstimatedStateBatch(
            pos_world=np.zeros((2, 3), dtype=np.float64),
            vel_world=np.zeros((2, 3), dtype=np.float64),
            quat_world_body=np.stack([_quat_from_rpy(0.0, 0.0, 0.0), _quat_from_rpy(0.0, 0.0, 0.2)], axis=0),
            omega_body=np.zeros((2, 3), dtype=np.float64),
        )
        traj = TrajectoryCommandBatch(
            pos_ref_world=np.array([[0.0, 0.0, 1.0], [0.2, -0.1, 1.2]], dtype=np.float64),
            vel_ref_world=np.zeros((2, 3), dtype=np.float64),
            acc_ref_world=np.zeros((2, 3), dtype=np.float64),
            yaw_ref=np.array([0.0, 0.1], dtype=np.float64),
            yaw_rate_ref=np.zeros((2,), dtype=np.float64),
        )

        state = stack.follower_step(state=state, traj=traj, est=est)
        out, _, _ = stack.controller_step(state=state, est=est)
        self.assertEqual(out.motor.shape, (2, 4))
        self.assertEqual(out.saturated.shape, (2,))


if __name__ == "__main__":
    unittest.main()
