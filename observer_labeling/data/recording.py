from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

from controller import (
    ControlStack,
    ControlStackConfig,
    ControlStackState,
    EstimatedStateBatch,
    TrajectoryCommandBatch,
)
from frames import ned_to_mj_world, yaw_ned_to_mj
from observer_labeling.data.dataset import TrajectoryDataset
from run_quadrotor import ACTUATOR_NAMES, BODY_NAME, MODEL_PATH, build_mixer
from trajgen import TrajectoryBuffer, generate_human_like_with_plan


ACCEL_SENSOR_NAME = "body_linacc"
MAG_DECLINATION_DEG = 10.6
MAG_INCLINATION_DEG = 65.0


def _normalize(vec: np.ndarray) -> np.ndarray:
    return vec / max(float(np.linalg.norm(vec)), 1e-9)


def _quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dtype=np.float64,
    )


def _quat_conj(q: np.ndarray) -> np.ndarray:
    return np.array([q[0], -q[1], -q[2], -q[3]], dtype=np.float64)


def _rotate_world_to_body(quat_wxyz: np.ndarray, vec_world: np.ndarray) -> np.ndarray:
    pure = np.array([0.0, vec_world[0], vec_world[1], vec_world[2]], dtype=np.float64)
    rotated = _quat_mul(_quat_mul(_quat_conj(quat_wxyz), pure), quat_wxyz)
    return rotated[1:]


def _magnetic_field_world() -> np.ndarray:
    decl = np.deg2rad(MAG_DECLINATION_DEG)
    incl = np.deg2rad(MAG_INCLINATION_DEG)
    mag_ned = np.array(
        [
            np.cos(incl) * np.cos(decl),
            np.cos(incl) * np.sin(decl),
            np.sin(incl),
        ],
        dtype=np.float64,
    )
    return _normalize(ned_to_mj_world(mag_ned[None, :])[0])


def _first_order_bias_step(
    bias: np.ndarray,
    target: np.ndarray,
    dt: float,
    tau: float,
    noise_std: float,
    rng: np.random.Generator,
) -> np.ndarray:
    alpha = np.clip(dt / max(tau, 1e-6), 0.0, 1.0)
    noise = rng.normal(scale=noise_std, size=3)
    return np.asarray((1.0 - alpha) * bias + alpha * target + noise, dtype=np.float64)


@dataclass(frozen=True)
class RecorderConfig:
    seed: int = 7
    duration: float = 25.0
    traj_dt: float = 0.05


def _load_model() -> tuple[mujoco.MjModel, mujoco.MjData, int, list[int], int, int]:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Expected Skydio X2 model at: {MODEL_PATH}. "
            "Clone MuJoCo Menagerie into ./mujoco_menagerie first."
        )

    config = ControlStackConfig()
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    model.opt.timestep = config.dt_inner
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)

    key_hover = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "hover")
    if key_hover >= 0:
        mujoco.mj_resetDataKeyframe(model, data, key_hover)
    mujoco.mj_forward(model, data)

    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, BODY_NAME)
    if body_id < 0:
        raise RuntimeError(f"Body '{BODY_NAME}' not found in model.")

    actuator_ids: list[int] = []
    for name in ACTUATOR_NAMES:
        actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        if actuator_id < 0:
            raise RuntimeError(f"Actuator '{name}' not found in model.")
        actuator_ids.append(actuator_id)

    gyro_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, "body_gyro")
    accel_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, ACCEL_SENSOR_NAME)
    if gyro_id < 0 or accel_id < 0:
        raise RuntimeError("Expected body_gyro and body_linacc sensors in the MuJoCo model.")

    return (
        model,
        data,
        body_id,
        actuator_ids,
        int(model.sensor_adr[gyro_id]),
        int(model.sensor_adr[accel_id]),
    )


def _load_human_like_trajectory(cfg: RecorderConfig) -> TrajectoryBuffer:
    traj, _ = generate_human_like_with_plan(seed=cfg.seed, dt=cfg.traj_dt, duration=cfg.duration)
    return traj


def record_human_like_dataset(cfg: RecorderConfig) -> TrajectoryDataset:
    trajectory = _load_human_like_trajectory(cfg)
    model, data, body_id, actuator_ids, gyro_adr, accel_adr = _load_model()

    ctrl_cfg = ControlStackConfig()
    mass = float(model.body_mass[body_id])
    mixer = build_mixer(model, actuator_ids)
    stack = ControlStack(cfg=ctrl_cfg, mixer=mixer, mass=mass)
    stack_state = ControlStackState.zeros(batch_size=1)
    outer_stride = int(round(ctrl_cfg.dt_outer / ctrl_cfg.dt_inner))
    final_time = float(trajectory.t[-1])
    max_total_thrust = float(np.sum(mixer.cmd_max))
    rng = np.random.default_rng(cfg.seed)
    mag_world = _magnetic_field_world()
    gyro_target = rng.uniform(-0.03, 0.03, size=3)
    accel_target = rng.uniform(-0.12, 0.12, size=3)
    mag_bias = rng.uniform(-0.001, 0.001, size=3)
    gyro_bias = np.zeros(3, dtype=np.float64)
    accel_bias = np.zeros(3, dtype=np.float64)

    t_log: list[float] = []
    cmd_rates_log: list[np.ndarray] = []
    throttle_log: list[np.ndarray] = []
    gyro_log: list[np.ndarray] = []
    accel_log: list[np.ndarray] = []
    mag_log: list[np.ndarray] = []
    quat_log: list[np.ndarray] = []
    gyro_bias_log: list[np.ndarray] = []
    accel_bias_log: list[np.ndarray] = []

    inner_idx = 0
    while data.time <= final_time:
        if inner_idx % outer_stride == 0:
            tq = np.array([data.time], dtype=np.float64)
            pos_ref_ned, vel_ref_ned, acc_ref_ned, yaw_ref_ned = trajectory.sample_batch(tq)
            traj_cmd_pos = ned_to_mj_world(pos_ref_ned)
            traj_cmd_vel = ned_to_mj_world(vel_ref_ned)
            traj_cmd_acc = ned_to_mj_world(acc_ref_ned)
            traj_cmd_yaw = yaw_ned_to_mj(yaw_ref_ned)

            est = EstimatedStateBatch(
                pos_world=data.qpos[0:3][None, :],
                vel_world=data.cvel[body_id, 3:6][None, :],
                quat_world_body=data.xquat[body_id][None, :],
                omega_body=data.sensordata[gyro_adr : gyro_adr + 3][None, :],
            )
            stack_state = stack.follower_step(
                state=stack_state,
                traj=TrajectoryCommandBatch(
                    pos_ref_world=traj_cmd_pos,
                    vel_ref_world=traj_cmd_vel,
                    acc_ref_world=traj_cmd_acc,
                    yaw_ref=traj_cmd_yaw,
                    yaw_rate_ref=np.zeros_like(traj_cmd_yaw),
                ),
                est=est,
            )

        est_inner = EstimatedStateBatch(
            pos_world=data.qpos[0:3][None, :],
            vel_world=data.cvel[body_id, 3:6][None, :],
            quat_world_body=data.xquat[body_id][None, :],
            omega_body=data.sensordata[gyro_adr : gyro_adr + 3][None, :],
        )
        motor_out, stack_state, _ = stack.controller_step(state=stack_state, est=est_inner)
        for i, actuator_id in enumerate(actuator_ids):
            data.ctrl[actuator_id] = float(motor_out.motor[0, i])

        cmd_proxy = np.array(
            [
                float(stack_state.last_command.cmd1[0]),
                float(stack_state.last_command.cmd2[0]),
                float(stack_state.last_command.cmd3[0]),
            ],
            dtype=np.float64,
        )
        throttle_proxy = np.array(
            [[np.clip(float(stack_state.last_command.cmd4[0]) / max(max_total_thrust, 1e-9), 0.0, 1.0)]],
            dtype=np.float64,
        )

        quat_world_body = np.asarray(data.xquat[body_id], dtype=np.float64)
        gyro_bias = _first_order_bias_step(gyro_bias, gyro_target, ctrl_cfg.dt_inner, 8.0, 3e-5, rng)
        accel_bias = _first_order_bias_step(accel_bias, accel_target, ctrl_cfg.dt_inner, 12.0, 8e-5, rng)
        gyro_meas = np.asarray(data.sensordata[gyro_adr : gyro_adr + 3], dtype=np.float64) + gyro_bias
        accel_meas = np.asarray(data.sensordata[accel_adr : accel_adr + 3], dtype=np.float64) + accel_bias
        mag_true_body = _rotate_world_to_body(quat_world_body, mag_world)
        mag_noise = rng.normal(scale=3e-4, size=3)
        mag_meas = _normalize(mag_true_body + mag_bias + mag_noise)

        t_log.append(float(data.time))
        cmd_rates_log.append(cmd_proxy)
        throttle_log.append(throttle_proxy[0].copy())
        gyro_log.append(gyro_meas.copy())
        accel_log.append(accel_meas.copy())
        mag_log.append(mag_meas.copy())
        quat_log.append(quat_world_body.copy())
        gyro_bias_log.append(gyro_bias.copy())
        accel_bias_log.append(accel_bias.copy())

        mujoco.mj_step(model, data)
        inner_idx += 1

    dataset = TrajectoryDataset(
        t=np.asarray(t_log, dtype=np.float64),
        cmd_rates=np.asarray(cmd_rates_log, dtype=np.float64),
        throttle=np.asarray(throttle_log, dtype=np.float64),
        gyro=np.asarray(gyro_log, dtype=np.float64),
        accel=np.asarray(accel_log, dtype=np.float64),
        mag_body=np.asarray(mag_log, dtype=np.float64),
        true_quat=np.asarray(quat_log, dtype=np.float64),
        true_bias=np.asarray(gyro_bias_log, dtype=np.float64),
        true_gyro_bias=np.asarray(gyro_bias_log, dtype=np.float64),
        true_accel_bias=np.asarray(accel_bias_log, dtype=np.float64),
        trajectory_name=f"human_like_seed_{cfg.seed}",
        cmd_rates_source="controller_command_proxy_roll_pitch_yawrate",
        estimator_rate_hz=1.0 / ctrl_cfg.dt_inner,
        decision_rate_hz=20.0,
        hold_steps=20,
        seed=cfg.seed,
    )
    dataset.validate()
    return dataset
