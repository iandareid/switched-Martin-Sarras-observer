"""Roscopter-style staged controller stack for quadrotor trajectory tracking.

Pipeline:
1) TrajectoryFollowerStage: trajectory + state -> roll/pitch/yaw-rate/thrust command
2) CascadedControllerStage: command + state -> motor commands through mixer

All arrays are batch-first to support GPU-friendly vectorization.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Protocol

import numpy as np


class ArrayBackend(Protocol):
    """Minimal array backend protocol for future GPU backends."""

    ndarray: type


class NumpyBackend:
    ndarray = np.ndarray


XP: ArrayBackend = NumpyBackend()


def wrap_angle(angle: np.ndarray) -> np.ndarray:
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def normalize(vec: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(vec, axis=-1, keepdims=True)
    return vec / np.maximum(n, 1e-9)


def quat_mul_wxyz(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]
    return np.stack(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        axis=-1,
    )


def quat_conj_wxyz(q: np.ndarray) -> np.ndarray:
    out = q.copy()
    out[..., 1:] *= -1.0
    return out


def rotmat_to_quat_wxyz(rot: np.ndarray) -> np.ndarray:
    q = np.zeros(rot.shape[:-2] + (4,), dtype=rot.dtype)
    tr = rot[..., 0, 0] + rot[..., 1, 1] + rot[..., 2, 2]

    m = tr > 0
    if np.any(m):
        s = np.sqrt(tr[m] + 1.0) * 2.0
        q[m, 0] = 0.25 * s
        q[m, 1] = (rot[m, 2, 1] - rot[m, 1, 2]) / s
        q[m, 2] = (rot[m, 0, 2] - rot[m, 2, 0]) / s
        q[m, 3] = (rot[m, 1, 0] - rot[m, 0, 1]) / s

    m0 = (~m) & (rot[..., 0, 0] > rot[..., 1, 1]) & (rot[..., 0, 0] > rot[..., 2, 2])
    if np.any(m0):
        s = np.sqrt(1.0 + rot[m0, 0, 0] - rot[m0, 1, 1] - rot[m0, 2, 2]) * 2.0
        q[m0, 0] = (rot[m0, 2, 1] - rot[m0, 1, 2]) / s
        q[m0, 1] = 0.25 * s
        q[m0, 2] = (rot[m0, 0, 1] + rot[m0, 1, 0]) / s
        q[m0, 3] = (rot[m0, 0, 2] + rot[m0, 2, 0]) / s

    m1 = (~m) & (~m0) & (rot[..., 1, 1] > rot[..., 2, 2])
    if np.any(m1):
        s = np.sqrt(1.0 + rot[m1, 1, 1] - rot[m1, 0, 0] - rot[m1, 2, 2]) * 2.0
        q[m1, 0] = (rot[m1, 0, 2] - rot[m1, 2, 0]) / s
        q[m1, 1] = (rot[m1, 0, 1] + rot[m1, 1, 0]) / s
        q[m1, 2] = 0.25 * s
        q[m1, 3] = (rot[m1, 1, 2] + rot[m1, 2, 1]) / s

    m2 = (~m) & (~m0) & (~m1)
    if np.any(m2):
        s = np.sqrt(1.0 + rot[m2, 2, 2] - rot[m2, 0, 0] - rot[m2, 1, 1]) * 2.0
        q[m2, 0] = (rot[m2, 1, 0] - rot[m2, 0, 1]) / s
        q[m2, 1] = (rot[m2, 0, 2] + rot[m2, 2, 0]) / s
        q[m2, 2] = (rot[m2, 1, 2] + rot[m2, 2, 1]) / s
        q[m2, 3] = 0.25 * s

    return normalize(q)


def quat_to_rpy(quat_wxyz: np.ndarray) -> np.ndarray:
    w, x, y, z = quat_wxyz[:, 0], quat_wxyz[:, 1], quat_wxyz[:, 2], quat_wxyz[:, 3]

    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    sinp = 2.0 * (w * y - z * x)
    pitch = np.arcsin(np.clip(sinp, -1.0, 1.0))

    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)

    return np.stack([roll, pitch, yaw], axis=1)


@dataclass
class PIDGains:
    kp: np.ndarray
    ki: np.ndarray
    kd: np.ndarray
    u_min: np.ndarray
    u_max: np.ndarray


@dataclass
class PIDState:
    integrator: np.ndarray
    differentiator: np.ndarray
    last_error: np.ndarray
    last_state: np.ndarray

    @classmethod
    def zeros(cls, batch_size: int, channels: int) -> "PIDState":
        shape = (batch_size, channels)
        return cls(
            integrator=np.zeros(shape, dtype=np.float64),
            differentiator=np.zeros(shape, dtype=np.float64),
            last_error=np.zeros(shape, dtype=np.float64),
            last_state=np.zeros(shape, dtype=np.float64),
        )


class BatchPID:
    def __init__(self, gains: PIDGains, channels: int) -> None:
        self.gains = gains
        self.channels = channels

    def step(
        self,
        state: PIDState,
        desired: np.ndarray,
        current: np.ndarray,
        dt: float,
        current_rate: np.ndarray | None = None,
        tau: float = 0.05,
    ) -> tuple[np.ndarray, PIDState]:
        if dt <= 1e-8:
            return np.zeros_like(desired), state

        error = desired - current

        p_term = self.gains.kp[None, :] * error

        if current_rate is not None:
            d_term = self.gains.kd[None, :] * current_rate
        else:
            alpha = (2.0 * tau - dt) / (2.0 * tau + dt)
            beta = 2.0 / (2.0 * tau + dt)
            state.differentiator = alpha * state.differentiator + beta * (current - state.last_state)
            d_term = self.gains.kd[None, :] * state.differentiator

        u_no_i = p_term - d_term
        u_sat = np.clip(u_no_i, self.gains.u_min[None, :], self.gains.u_max[None, :])
        unsat = np.isclose(u_no_i, u_sat)

        integrate_mask = unsat & (self.gains.ki[None, :] > 0.0)
        state.integrator = np.where(
            integrate_mask,
            state.integrator + 0.5 * dt * (error + state.last_error),
            state.integrator,
        )

        i_term = self.gains.ki[None, :] * state.integrator
        u = p_term + i_term - d_term
        u_clip = np.clip(u, self.gains.u_min[None, :], self.gains.u_max[None, :])

        can_backcalc = self.gains.ki[None, :] > 0.0
        needed_i = (u_clip - p_term + d_term) / np.maximum(self.gains.ki[None, :], 1e-9)
        state.integrator = np.where((~np.isclose(u, u_clip)) & can_backcalc, needed_i, state.integrator)

        state.last_error = error
        state.last_state = current
        return u_clip, state


class ControlMode(IntEnum):
    ROLL_PITCH_YAWRATE_THRUST_TO_MIXER = 8
    ROLLRATE_PITCHRATE_YAWRATE_THRUST_TO_MIXER = 9
    PASS_THROUGH_TO_MIXER = 10


@dataclass
class TrajectoryCommandBatch:
    pos_ref_world: np.ndarray
    vel_ref_world: np.ndarray
    acc_ref_world: np.ndarray
    yaw_ref: np.ndarray
    yaw_rate_ref: np.ndarray


@dataclass
class EstimatedStateBatch:
    pos_world: np.ndarray
    vel_world: np.ndarray
    quat_world_body: np.ndarray
    omega_body: np.ndarray


@dataclass
class ControllerCommandBatch:
    mode: np.ndarray
    cmd1: np.ndarray
    cmd2: np.ndarray
    cmd3: np.ndarray
    cmd4: np.ndarray


@dataclass
class MotorCommandBatch:
    motor: np.ndarray
    saturated: np.ndarray


@dataclass
class FollowerState:
    pos_pid: PIDState
    yaw_pid: PIDState

    @classmethod
    def zeros(cls, batch_size: int) -> "FollowerState":
        return cls(pos_pid=PIDState.zeros(batch_size, 3), yaw_pid=PIDState.zeros(batch_size, 1))


@dataclass
class ControllerState:
    angle_pid: PIDState
    rate_pid: PIDState

    @classmethod
    def zeros(cls, batch_size: int) -> "ControllerState":
        return cls(angle_pid=PIDState.zeros(batch_size, 3), rate_pid=PIDState.zeros(batch_size, 3))


@dataclass
class TrajectoryFollowerConfig:
    gravity: float = 9.81
    max_tilt_rad: float = np.deg2rad(45.0)
    down_command_window: float = 3.0
    max_commanded_down_accel_in_gs: float = -0.4
    tau: float = 0.05
    pos_gains: PIDGains = field(
        default_factory=lambda: PIDGains(
            kp=np.array([4.0, 4.0, 4.0], dtype=np.float64),
            ki=np.array([0.01, 0.01, 0.05], dtype=np.float64),
            kd=np.array([2.0, 2.0, 3.5], dtype=np.float64),
            u_min=np.array([-30.0, -30.0, -30.0], dtype=np.float64),
            u_max=np.array([30.0, 30.0, 30.0], dtype=np.float64),
        )
    )
    yaw_gains: PIDGains = field(
        default_factory=lambda: PIDGains(
            kp=np.array([2.0], dtype=np.float64),
            ki=np.array([0.0], dtype=np.float64),
            kd=np.array([1.0], dtype=np.float64),
            u_min=np.array([-4.0], dtype=np.float64),
            u_max=np.array([4.0], dtype=np.float64),
        )
    )


@dataclass
class CascadedControllerConfig:
    tau: float = 0.05
    max_roll_rad: float = np.deg2rad(45.0)
    max_pitch_rad: float = np.deg2rad(45.0)
    max_yaw_rate: float = np.deg2rad(120.0)
    angle_gains: PIDGains = field(
        default_factory=lambda: PIDGains(
            kp=np.array([5.0, 5.0, 2.0], dtype=np.float64),
            ki=np.array([0.0, 0.0, 0.002], dtype=np.float64),
            kd=np.array([0.8, 0.8, 1.0], dtype=np.float64),
            u_min=np.array([-10.0, -10.0, -10.0], dtype=np.float64),
            u_max=np.array([10.0, 10.0, 10.0], dtype=np.float64),
        )
    )
    rate_gains: PIDGains = field(
        default_factory=lambda: PIDGains(
            kp=np.array([1.0, 1.0, 1.0], dtype=np.float64),
            ki=np.array([0.0, 0.0, 0.0], dtype=np.float64),
            kd=np.array([0.0, 0.0, 0.0], dtype=np.float64),
            u_min=np.array([-10.0, -10.0, -10.0], dtype=np.float64),
            u_max=np.array([10.0, 10.0, 10.0], dtype=np.float64),
        )
    )


@dataclass
class Mixer:
    alloc: np.ndarray
    inv_alloc: np.ndarray
    cmd_min: np.ndarray
    cmd_max: np.ndarray

    @classmethod
    def from_geometry(
        cls,
        rotor_pos_body: np.ndarray,
        yaw_coeff: np.ndarray,
        cmd_min: np.ndarray,
        cmd_max: np.ndarray,
    ) -> "Mixer":
        x = rotor_pos_body[:, 0]
        y = rotor_pos_body[:, 1]
        alloc = np.array(
            [
                np.ones(rotor_pos_body.shape[0], dtype=np.float64),
                y,
                -x,
                yaw_coeff,
            ],
            dtype=np.float64,
        )
        inv_alloc = np.linalg.pinv(alloc)
        return cls(alloc=alloc, inv_alloc=inv_alloc, cmd_min=cmd_min, cmd_max=cmd_max)

    def mix(self, thrust: np.ndarray, tau_body: np.ndarray) -> MotorCommandBatch:
        wrench = np.concatenate([thrust[:, None], tau_body], axis=1)
        cmd = wrench @ self.inv_alloc.T
        clipped = np.clip(cmd, self.cmd_min[None, :], self.cmd_max[None, :])
        saturated = np.any(np.abs(clipped - cmd) > 1e-8, axis=1)
        return MotorCommandBatch(motor=clipped, saturated=saturated)


def _compute_desired_quat_and_thrust(
    accel_cmd_world: np.ndarray,
    yaw_ref: np.ndarray,
    gravity: float,
    max_tilt_rad: float,
    mass: float,
) -> tuple[np.ndarray, np.ndarray]:
    g_vec = np.array([0.0, 0.0, gravity], dtype=np.float64)
    thrust_vec = mass * (accel_cmd_world + g_vec[None, :])

    b3 = normalize(thrust_vec)
    z = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    tilt = np.arccos(np.clip(np.sum(b3 * z[None, :], axis=1), -1.0, 1.0))
    tilt_limited = np.minimum(tilt, max_tilt_rad)

    lateral = b3[:, :2]
    lateral_norm = np.linalg.norm(lateral, axis=1, keepdims=True)
    lateral_dir = lateral / np.maximum(lateral_norm, 1e-9)
    b3 = np.concatenate([lateral_dir * np.sin(tilt_limited)[:, None], np.cos(tilt_limited)[:, None]], axis=1)
    b3 = normalize(b3)

    x_c = np.stack([np.cos(yaw_ref), np.sin(yaw_ref), np.zeros_like(yaw_ref)], axis=1)
    b2 = normalize(np.cross(b3, x_c))
    b1 = normalize(np.cross(b2, b3))
    rot = np.stack([b1, b2, b3], axis=2)

    thrust = np.maximum(np.sum(thrust_vec * b3, axis=1), 0.0)
    quat = rotmat_to_quat_wxyz(rot)
    return quat, thrust


class TrajectoryFollowerStage:
    def __init__(self, cfg: TrajectoryFollowerConfig) -> None:
        self.cfg = cfg
        self.pos_pid = BatchPID(cfg.pos_gains, channels=3)
        self.yaw_pid = BatchPID(cfg.yaw_gains, channels=1)

    def step(
        self,
        state: FollowerState,
        traj: TrajectoryCommandBatch,
        est: EstimatedStateBatch,
        dt: float,
        mass: float,
    ) -> tuple[ControllerCommandBatch, FollowerState, dict[str, np.ndarray]]:
        pos_err_down = traj.pos_ref_world[:, 2] - est.pos_world[:, 2]
        clamped_ref_z = np.where(
            pos_err_down < -self.cfg.down_command_window,
            est.pos_world[:, 2] - self.cfg.down_command_window,
            traj.pos_ref_world[:, 2],
        )
        pos_ref = traj.pos_ref_world.copy()
        pos_ref[:, 2] = clamped_ref_z

        u_tilde, state.pos_pid = self.pos_pid.step(
            state.pos_pid,
            desired=pos_ref,
            current=est.pos_world,
            dt=dt,
            current_rate=est.vel_world,
            tau=self.cfg.tau,
        )

        yaw_des = traj.yaw_ref[:, None]
        yaw_now = quat_to_rpy(est.quat_world_body)[:, 2:3]
        yaw_wrapped = yaw_now + wrap_angle(yaw_des - yaw_now)
        yaw_rate_cmd, state.yaw_pid = self.yaw_pid.step(
            state.yaw_pid,
            desired=yaw_wrapped,
            current=yaw_now,
            dt=dt,
            current_rate=est.omega_body[:, 2:3],
            tau=self.cfg.tau,
        )

        accel_cmd = u_tilde + traj.acc_ref_world
        accel_cmd[:, 2] = np.maximum(
            accel_cmd[:, 2],
            self.cfg.max_commanded_down_accel_in_gs * self.cfg.gravity,
        )

        quat_des, thrust_cmd = _compute_desired_quat_and_thrust(
            accel_cmd_world=accel_cmd,
            yaw_ref=traj.yaw_ref,
            gravity=self.cfg.gravity,
            max_tilt_rad=self.cfg.max_tilt_rad,
            mass=mass,
        )
        rpy_des = quat_to_rpy(quat_des)

        cmd = ControllerCommandBatch(
            mode=np.full((traj.pos_ref_world.shape[0],), ControlMode.ROLL_PITCH_YAWRATE_THRUST_TO_MIXER, dtype=np.int64),
            cmd1=np.clip(rpy_des[:, 0], -self.cfg.max_tilt_rad, self.cfg.max_tilt_rad),
            cmd2=np.clip(rpy_des[:, 1], -self.cfg.max_tilt_rad, self.cfg.max_tilt_rad),
            cmd3=(yaw_rate_cmd[:, 0] + traj.yaw_rate_ref),
            cmd4=thrust_cmd,
        )
        dbg = {
            "quat_des": quat_des,
            "thrust_cmd": thrust_cmd,
            "yaw_rate_cmd": cmd.cmd3,
            "accel_cmd": accel_cmd,
        }
        return cmd, state, dbg


class CascadedControllerStage:
    def __init__(self, cfg: CascadedControllerConfig, mixer: Mixer) -> None:
        self.cfg = cfg
        self.mixer = mixer
        self.angle_pid = BatchPID(cfg.angle_gains, channels=3)
        self.rate_pid = BatchPID(cfg.rate_gains, channels=3)

    def step(
        self,
        state: ControllerState,
        cmd: ControllerCommandBatch,
        est: EstimatedStateBatch,
        dt: float,
    ) -> tuple[MotorCommandBatch, ControllerState, dict[str, np.ndarray]]:
        rpy = quat_to_rpy(est.quat_world_body)
        mode = cmd.mode

        phi = np.clip(cmd.cmd1, -self.cfg.max_roll_rad, self.cfg.max_roll_rad)
        theta = np.clip(cmd.cmd2, -self.cfg.max_pitch_rad, self.cfg.max_pitch_rad)
        yaw_rate = np.clip(cmd.cmd3, -self.cfg.max_yaw_rate, self.cfg.max_yaw_rate)
        thrust = np.maximum(cmd.cmd4, 0.0)

        tau_body = np.zeros((cmd.cmd1.shape[0], 3), dtype=np.float64)

        mask_angle = mode == int(ControlMode.ROLL_PITCH_YAWRATE_THRUST_TO_MIXER)
        if np.any(mask_angle):
            desired = np.stack([phi, theta, np.zeros_like(phi)], axis=1)
            current = np.stack([rpy[:, 0], rpy[:, 1], np.zeros_like(phi)], axis=1)
            rates = np.stack([est.omega_body[:, 0], est.omega_body[:, 1], est.omega_body[:, 2]], axis=1)

            tau_angle, state.angle_pid = self.angle_pid.step(
                state.angle_pid,
                desired=desired,
                current=current,
                dt=dt,
                current_rate=rates,
                tau=self.cfg.tau,
            )
            yaw_u, state.rate_pid = self.rate_pid.step(
                state.rate_pid,
                desired=np.stack([np.zeros_like(yaw_rate), np.zeros_like(yaw_rate), yaw_rate], axis=1),
                current=np.stack([np.zeros_like(yaw_rate), np.zeros_like(yaw_rate), est.omega_body[:, 2]], axis=1),
                dt=dt,
                current_rate=None,
                tau=self.cfg.tau,
            )
            tau_body = tau_angle
            tau_body[:, 2] = yaw_u[:, 2]

        mask_rate = mode == int(ControlMode.ROLLRATE_PITCHRATE_YAWRATE_THRUST_TO_MIXER)
        if np.any(mask_rate):
            desired_rate = np.stack([phi, theta, yaw_rate], axis=1)
            tau_rate, state.rate_pid = self.rate_pid.step(
                state.rate_pid,
                desired=desired_rate,
                current=est.omega_body,
                dt=dt,
                current_rate=None,
                tau=self.cfg.tau,
            )
            tau_body = np.where(mask_rate[:, None], tau_rate, tau_body)

        mask_passthrough = mode == int(ControlMode.PASS_THROUGH_TO_MIXER)
        if np.any(mask_passthrough):
            tau_passthrough = np.stack([cmd.cmd1, cmd.cmd2, cmd.cmd3], axis=1)
            tau_body = np.where(mask_passthrough[:, None], tau_passthrough, tau_body)

        motor = self.mixer.mix(thrust=thrust, tau_body=tau_body)
        dbg = {"tau_body": tau_body, "thrust": thrust, "rpy": rpy}
        return motor, state, dbg


@dataclass
class ControlStackConfig:
    dt_inner: float = 1.0 / 400.0
    dt_outer: float = 1.0 / 100.0
    follower: TrajectoryFollowerConfig = field(default_factory=TrajectoryFollowerConfig)
    controller: CascadedControllerConfig = field(default_factory=CascadedControllerConfig)


@dataclass
class ControlStackState:
    follower: FollowerState
    controller: ControllerState
    last_command: ControllerCommandBatch
    last_debug: dict[str, np.ndarray]

    @classmethod
    def zeros(cls, batch_size: int) -> "ControlStackState":
        zero = np.zeros((batch_size,), dtype=np.float64)
        cmd = ControllerCommandBatch(
            mode=np.full((batch_size,), int(ControlMode.ROLL_PITCH_YAWRATE_THRUST_TO_MIXER), dtype=np.int64),
            cmd1=zero.copy(),
            cmd2=zero.copy(),
            cmd3=zero.copy(),
            cmd4=zero.copy(),
        )
        return cls(
            follower=FollowerState.zeros(batch_size),
            controller=ControllerState.zeros(batch_size),
            last_command=cmd,
            last_debug={},
        )


class ControlStack:
    def __init__(self, cfg: ControlStackConfig, mixer: Mixer, mass: float) -> None:
        self.cfg = cfg
        self.mass = mass
        self.follower = TrajectoryFollowerStage(cfg.follower)
        self.controller = CascadedControllerStage(cfg.controller, mixer)

    def follower_step(
        self,
        state: ControlStackState,
        traj: TrajectoryCommandBatch,
        est: EstimatedStateBatch,
    ) -> ControlStackState:
        cmd, follower_state, dbg = self.follower.step(
            state=state.follower,
            traj=traj,
            est=est,
            dt=self.cfg.dt_outer,
            mass=self.mass,
        )
        state.follower = follower_state
        state.last_command = cmd
        state.last_debug = dbg
        return state

    def controller_step(
        self,
        state: ControlStackState,
        est: EstimatedStateBatch,
    ) -> tuple[MotorCommandBatch, ControlStackState, dict[str, np.ndarray]]:
        motor, ctrl_state, dbg = self.controller.step(
            state=state.controller,
            cmd=state.last_command,
            est=est,
            dt=self.cfg.dt_inner,
        )
        state.controller = ctrl_state
        dbg_all = dict(state.last_debug)
        dbg_all.update(dbg)
        return motor, state, dbg_all
