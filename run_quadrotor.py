#!/usr/bin/env python3
"""Run Skydio X2 with cascaded PID trajectory tracking."""

from __future__ import annotations

import argparse
from collections import deque
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

from controller import (
    ControlStack,
    ControlStackConfig,
    ControlStackState,
    EstimatedStateBatch,
    Mixer,
    TrajectoryCommandBatch,
)
from frames import ned_to_mj_world, yaw_ned_to_mj
from trajgen import TrajectoryBuffer, generate_default_demo, generate_human_like_with_plan
from trajgen.profiles import HumanLikeSegment


MODEL_PATH = Path(__file__).with_name("mujoco_menagerie") / "skydio_x2" / "scene.xml"
BODY_NAME = "x2"
ACTUATOR_NAMES = ("thrust1", "thrust2", "thrust3", "thrust4")
GYRO_SENSOR_NAME = "body_gyro"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    traj_group = parser.add_mutually_exclusive_group()
    traj_group.add_argument(
        "--trajectory",
        type=Path,
        default=None,
        help="Optional CSV with columns: t,x,y,z,yaw (NED frame).",
    )
    traj_group.add_argument(
        "--human-like",
        action="store_true",
        help="Generate and track a seeded human-like trajectory.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed used by --human-like trajectory generation.",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=20.0,
        help="Duration in seconds for --human-like trajectory generation.",
    )
    parser.add_argument(
        "--traj-dt",
        type=float,
        default=0.05,
        help="Sampling period in seconds for generated trajectories.",
    )
    parser.add_argument(
        "--traj-diagnostics",
        action="store_true",
        help="Print generated trajectory segment diagnostics to stdout.",
    )
    parser.add_argument(
        "--traj-diagnostics-live",
        action="store_true",
        help="With --traj-diagnostics, update plots live during simulation (default is plot after close).",
    )
    parser.add_argument(
        "--controller-diagnostics",
        action="store_true",
        help="Print controller metrics during flight (tracking error, thrust, saturation rate).",
    )
    parser.add_argument(
        "--no-realtime",
        action="store_true",
        help="Run simulation as fast as possible.",
    )
    return parser.parse_args()


def _print_human_like_plan(seed: int, segments: list[HumanLikeSegment]) -> None:
    print(f"[traj] human-like seed={seed}, segments={len(segments)}")
    for i, seg in enumerate(segments):
        p0 = ", ".join(f"{v: .2f}" for v in seg.start_pos)
        p1 = ", ".join(f"{v: .2f}" for v in seg.end_pos)
        print(
            f"[traj] {i:02d} {seg.kind:11s} "
            f"t=[{seg.t_start:6.2f}, {seg.t_end:6.2f}] "
            f"pos0=[{p0}] pos1=[{p1}] "
            f"yaw0={seg.start_yaw: .2f} yaw1={seg.end_yaw: .2f}"
        )


def _quat_wxyz_to_yaw(quat_wxyz: np.ndarray) -> float:
    w, x, y, z = quat_wxyz
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return float(np.arctan2(siny_cosp, cosy_cosp))


def _quat_wxyz_to_rpy(quat_wxyz: np.ndarray) -> np.ndarray:
    w, x, y, z = quat_wxyz

    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    sinp = 2.0 * (w * y - z * x)
    pitch = np.arcsin(np.clip(sinp, -1.0, 1.0))

    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)

    return np.array([roll, pitch, yaw], dtype=np.float64)


def _wrap_pi(angle: np.ndarray) -> np.ndarray:
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


class LiveDiagnosticsPlotter:
    def __init__(self, update_hz: float = 10.0) -> None:
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            self.enabled = False
            self._plt = None
            print("[traj] matplotlib not installed; run `pip install matplotlib` to enable plots")
            return

        self.enabled = True
        self._plt = plt
        self._update_period_s = 1.0 / max(update_hz, 1e-6)
        self._last_update_wall = 0.0

        plt.ion()
        fig, axs = plt.subplots(3, 2, figsize=(13, 10), sharex=True)
        self._fig = fig
        self._axs = axs
        self._fig.suptitle("Trajectory Diagnostics: Commands vs Responses")

        self._lines: dict[str, object] = {}
        labels = ("x", "y", "z")
        for i, lbl in enumerate(labels):
            self._lines[f"pos_ref_{i}"] = axs[0, 0].plot([], [], "--", label=f"{lbl}_ref")[0]
            self._lines[f"pos_rsp_{i}"] = axs[0, 0].plot([], [], label=f"{lbl}_rsp")[0]
            self._lines[f"vel_ref_{i}"] = axs[1, 0].plot([], [], "--", label=f"{lbl}_ref")[0]
            self._lines[f"vel_rsp_{i}"] = axs[1, 0].plot([], [], label=f"{lbl}_rsp")[0]

        self._lines["pos_err"] = axs[2, 0].plot([], [], label="|pos_err|")[0]
        self._lines["vel_err"] = axs[2, 0].plot([], [], label="|vel_err|")[0]
        self._lines["roll_ref"] = axs[0, 1].plot([], [], "--", label="roll_ref")[0]
        self._lines["roll_rsp"] = axs[0, 1].plot([], [], label="roll_rsp")[0]
        self._lines["pitch_ref"] = axs[0, 1].plot([], [], "--", label="pitch_ref")[0]
        self._lines["pitch_rsp"] = axs[0, 1].plot([], [], label="pitch_rsp")[0]
        self._lines["yaw_ref"] = axs[0, 1].plot([], [], "--", label="yaw_ref")[0]
        self._lines["yaw_rsp"] = axs[0, 1].plot([], [], label="yaw_rsp")[0]
        self._lines["omega_z_ref"] = axs[1, 1].plot([], [], "--", label="omega_z_ref")[0]
        self._lines["omega_z_rsp"] = axs[1, 1].plot([], [], label="omega_z_rsp")[0]
        self._lines["thrust_cmd"] = axs[2, 1].plot([], [], label="thrust_cmd")[0]
        self._lines["sat_rate_1s"] = axs[2, 1].plot([], [], label="sat_rate_1s")[0]

        axs[0, 0].set_ylabel("Position (m)")
        axs[0, 0].set_title("Position")
        axs[1, 0].set_ylabel("Velocity (m/s)")
        axs[1, 0].set_title("Velocity")
        axs[2, 0].set_ylabel("Error Norm")
        axs[2, 0].set_xlabel("Time (s)")
        axs[2, 0].set_title("Tracking Error")
        axs[0, 1].set_ylabel("Angle (rad)")
        axs[0, 1].set_title("Attitude (RPY)")
        axs[0, 1].set_ylim(-np.pi, np.pi)
        axs[1, 1].set_ylabel("Yaw Rate (rad/s)")
        axs[1, 1].set_title("Yaw Rate")
        axs[2, 1].set_ylabel("Cmd / Rate")
        axs[2, 1].set_xlabel("Time (s)")
        axs[2, 1].set_title("Collective Thrust + Saturation")

        for ax in axs.flat:
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8, ncol=3 if ax in (axs[0, 0], axs[1, 0]) else 1)
        self._plt.tight_layout()
        self._fig.canvas.draw_idle()
        self._fig.canvas.flush_events()

    def maybe_update(self, log: dict[str, list[np.ndarray | float]], *, force: bool = False) -> None:
        if not self.enabled or not log["t"]:
            return

        now = time.perf_counter()
        if not force and now - self._last_update_wall < self._update_period_s:
            return

        t = np.asarray(log["t"], dtype=np.float64)
        pos_ref = np.asarray(log["pos_ref"], dtype=np.float64)
        pos_rsp = np.asarray(log["pos_rsp"], dtype=np.float64)
        vel_ref = np.asarray(log["vel_ref"], dtype=np.float64)
        vel_rsp = np.asarray(log["vel_rsp"], dtype=np.float64)
        yaw_ref = _wrap_pi(np.asarray(log["yaw_ref"], dtype=np.float64))
        yaw_rsp = _wrap_pi(np.asarray(log["yaw_rsp"], dtype=np.float64))
        roll_ref = _wrap_pi(np.asarray(log["roll_ref"], dtype=np.float64))
        roll_rsp = _wrap_pi(np.asarray(log["roll_rsp"], dtype=np.float64))
        pitch_ref = _wrap_pi(np.asarray(log["pitch_ref"], dtype=np.float64))
        pitch_rsp = _wrap_pi(np.asarray(log["pitch_rsp"], dtype=np.float64))
        omega_z_ref = np.asarray(log["omega_z_ref"], dtype=np.float64)
        omega_z_rsp = np.asarray(log["omega_z_rsp"], dtype=np.float64)
        thrust_cmd = np.asarray(log["thrust_cmd"], dtype=np.float64)
        sat_rate = np.asarray(log["sat_rate_1s"], dtype=np.float64)
        pos_err = np.linalg.norm(pos_ref - pos_rsp, axis=1)
        vel_err = np.linalg.norm(vel_ref - vel_rsp, axis=1)

        for i in range(3):
            self._lines[f"pos_ref_{i}"].set_data(t, pos_ref[:, i])
            self._lines[f"pos_rsp_{i}"].set_data(t, pos_rsp[:, i])
            self._lines[f"vel_ref_{i}"].set_data(t, vel_ref[:, i])
            self._lines[f"vel_rsp_{i}"].set_data(t, vel_rsp[:, i])

        self._lines["pos_err"].set_data(t, pos_err)
        self._lines["vel_err"].set_data(t, vel_err)
        self._lines["roll_ref"].set_data(t, roll_ref)
        self._lines["roll_rsp"].set_data(t, roll_rsp)
        self._lines["pitch_ref"].set_data(t, pitch_ref)
        self._lines["pitch_rsp"].set_data(t, pitch_rsp)
        self._lines["yaw_ref"].set_data(t, yaw_ref)
        self._lines["yaw_rsp"].set_data(t, yaw_rsp)
        self._lines["omega_z_ref"].set_data(t, omega_z_ref)
        self._lines["omega_z_rsp"].set_data(t, omega_z_rsp)
        self._lines["thrust_cmd"].set_data(t, thrust_cmd)
        self._lines["sat_rate_1s"].set_data(t, sat_rate)

        for ax in self._axs.flat:
            if ax is self._axs[0, 1]:
                continue
            ax.relim()
            ax.autoscale_view()
        self._axs[0, 1].set_ylim(-np.pi, np.pi)

        self._fig.canvas.draw_idle()
        self._fig.canvas.flush_events()
        self._plt.pause(0.001)
        self._last_update_wall = now

    def finalize(self, log: dict[str, list[np.ndarray | float]]) -> None:
        if not self.enabled:
            return
        self.maybe_update(log, force=True)
        self._plt.ioff()
        self._plt.show()


def load_trajectory(args: argparse.Namespace) -> tuple[TrajectoryBuffer, list[HumanLikeSegment] | None]:
    if args.human_like:
        if args.duration <= 0.0:
            raise ValueError("--duration must be > 0 for --human-like.")
        if args.traj_dt <= 0.0:
            raise ValueError("--traj-dt must be > 0 for --human-like.")
        traj, segments = generate_human_like_with_plan(
            seed=args.seed, dt=args.traj_dt, duration=args.duration
        )
        if args.traj_diagnostics:
            _print_human_like_plan(seed=args.seed, segments=segments)
        return traj, segments

    path = args.trajectory
    if path is None:
        if args.traj_dt <= 0.0:
            raise ValueError("--traj-dt must be > 0.")
        return generate_default_demo(dt=args.traj_dt), None
    if not path.exists():
        raise FileNotFoundError(f"Trajectory file not found: {path}")
    return TrajectoryBuffer.from_csv(path), None


def build_mixer(model: mujoco.MjModel, actuator_ids: list[int]) -> Mixer:
    rotor_pos = []
    yaw_coeff = []
    cmd_min = []
    cmd_max = []
    for actuator_id in actuator_ids:
        site_id = int(model.actuator_trnid[actuator_id, 0])
        rotor_pos.append(model.site_pos[site_id].copy())
        yaw_coeff.append(float(model.actuator_gear[actuator_id, 5]))
        low, high = model.actuator_ctrlrange[actuator_id]
        cmd_min.append(float(low))
        cmd_max.append(float(high))

    return Mixer.from_geometry(
        rotor_pos_body=np.asarray(rotor_pos, dtype=np.float64),
        yaw_coeff=np.asarray(yaw_coeff, dtype=np.float64),
        cmd_min=np.asarray(cmd_min, dtype=np.float64),
        cmd_max=np.asarray(cmd_max, dtype=np.float64),
    )


def main() -> None:
    args = parse_args()
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Expected Skydio X2 model at: {MODEL_PATH}. "
            "Clone MuJoCo Menagerie into ./mujoco_menagerie first."
        )

    trajectory, trajectory_segments = load_trajectory(args)
    config = ControlStackConfig()

    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    model.opt.timestep = config.dt_inner
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)

    key_hover = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "hover")
    if key_hover >= 0:
        mujoco.mj_resetDataKeyframe(model, data, key_hover)

    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, BODY_NAME)
    if body_id < 0:
        raise RuntimeError(f"Body '{BODY_NAME}' not found in model.")
    mass = float(model.body_mass[body_id])

    actuator_ids: list[int] = []
    for name in ACTUATOR_NAMES:
        actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        if actuator_id < 0:
            raise RuntimeError(f"Actuator '{name}' not found in model.")
        actuator_ids.append(actuator_id)

    gyro_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, GYRO_SENSOR_NAME)
    if gyro_id < 0:
        raise RuntimeError(f"Sensor '{GYRO_SENSOR_NAME}' not found in model.")
    gyro_adr = int(model.sensor_adr[gyro_id])

    mixer = build_mixer(model, actuator_ids)
    stack = ControlStack(cfg=config, mixer=mixer, mass=mass)
    stack_state = ControlStackState.zeros(batch_size=1)
    outer_stride = int(round(config.dt_outer / config.dt_inner))
    if outer_stride < 1:
        raise RuntimeError("Invalid loop rates: dt_outer must be >= dt_inner.")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.distance = 3.2
        viewer.cam.azimuth = 120
        viewer.cam.elevation = -25
        viewer.cam.lookat[:] = (0.0, 0.0, 0.4)

        wall_start = time.perf_counter()
        sim_start = data.time
        inner_idx = 0
        segment_idx = -1
        sat_window_len = max(1, int(round(1.0 / config.dt_inner)))
        sat_window: deque[float] = deque(maxlen=sat_window_len)
        prev_yaw_ref = np.array([0.0], dtype=np.float64)
        last_pos_err_norm = 0.0
        last_vel_err_norm = 0.0
        diag_log: dict[str, list[np.ndarray | float]] = {
            "t": [],
            "pos_ref": [],
            "pos_rsp": [],
            "vel_ref": [],
            "vel_rsp": [],
            "yaw_ref": [],
            "yaw_rsp": [],
            "roll_ref": [],
            "roll_rsp": [],
            "pitch_ref": [],
            "pitch_rsp": [],
            "omega_z_ref": [],
            "omega_z_rsp": [],
            "thrust_cmd": [],
            "sat_rate_1s": [],
        }
        live_plotter = (
            LiveDiagnosticsPlotter(update_hz=10.0)
            if args.traj_diagnostics and args.traj_diagnostics_live
            else None
        )

        while viewer.is_running():
            if args.traj_diagnostics and trajectory_segments:
                while (
                    segment_idx + 1 < len(trajectory_segments)
                    and data.time >= trajectory_segments[segment_idx + 1].t_start
                ):
                    segment_idx += 1
                    seg = trajectory_segments[segment_idx]
                    print(
                        f"[traj] entering segment {segment_idx:02d} ({seg.kind}) "
                        f"at sim t={data.time:.2f}s"
                    )

            if inner_idx % outer_stride == 0:
                tq = np.array([data.time], dtype=np.float64)
                pos_ref_ned, vel_ref_ned, acc_ref_ned, yaw_ref_ned = trajectory.sample_batch(tq)
                pos_ref = ned_to_mj_world(pos_ref_ned)
                vel_ref = ned_to_mj_world(vel_ref_ned)
                acc_ref = ned_to_mj_world(acc_ref_ned)
                yaw_ref = yaw_ned_to_mj(yaw_ref_ned)

                pos = data.qpos[0:3][None, :]
                # Use body COM linear velocity in world coordinates.
                vel = data.cvel[body_id, 3:6][None, :]
                last_pos_err_norm = float(np.linalg.norm(pos_ref - pos))
                last_vel_err_norm = float(np.linalg.norm(vel_ref - vel))

                yaw_rate_ref = np.zeros_like(yaw_ref)
                if inner_idx > 0:
                    yaw_rate_ref = _wrap_pi(yaw_ref - prev_yaw_ref) / config.dt_outer
                prev_yaw_ref = yaw_ref.copy()

                est = EstimatedStateBatch(
                    pos_world=pos,
                    vel_world=vel,
                    quat_world_body=data.xquat[body_id][None, :],
                    omega_body=data.sensordata[gyro_adr : gyro_adr + 3][None, :],
                )
                traj_cmd = TrajectoryCommandBatch(
                    pos_ref_world=pos_ref,
                    vel_ref_world=vel_ref,
                    acc_ref_world=acc_ref,
                    yaw_ref=yaw_ref,
                    yaw_rate_ref=yaw_rate_ref,
                )
                stack_state = stack.follower_step(
                    state=stack_state,
                    traj=traj_cmd,
                    est=est,
                )

                if args.traj_diagnostics:
                    rpy_ref = _quat_wxyz_to_rpy(stack_state.last_debug["quat_des"][0])
                    rpy_rsp = _quat_wxyz_to_rpy(data.xquat[body_id])
                    yaw_rsp = float(rpy_rsp[2])
                    sat_rate = float(np.mean(sat_window)) if sat_window else 0.0
                    diag_log["t"].append(float(data.time))
                    diag_log["pos_ref"].append(pos_ref[0].copy())
                    diag_log["pos_rsp"].append(pos[0].copy())
                    diag_log["vel_ref"].append(vel_ref[0].copy())
                    diag_log["vel_rsp"].append(vel[0].copy())
                    diag_log["yaw_ref"].append(float(yaw_ref[0]))
                    diag_log["yaw_rsp"].append(yaw_rsp)
                    diag_log["roll_ref"].append(float(rpy_ref[0]))
                    diag_log["roll_rsp"].append(float(rpy_rsp[0]))
                    diag_log["pitch_ref"].append(float(rpy_ref[1]))
                    diag_log["pitch_rsp"].append(float(rpy_rsp[1]))
                    diag_log["omega_z_ref"].append(float(stack_state.last_debug["yaw_rate_cmd"][0]))
                    diag_log["omega_z_rsp"].append(float(data.sensordata[gyro_adr + 2]))
                    diag_log["thrust_cmd"].append(float(stack_state.last_debug["thrust_cmd"][0]))
                    diag_log["sat_rate_1s"].append(float(sat_rate))
                    if live_plotter is not None:
                        live_plotter.maybe_update(diag_log)

                if args.controller_diagnostics:
                    sat_rate = float(np.mean(sat_window)) if sat_window else 0.0
                    print(
                        f"[ctrl] t={data.time:6.2f} "
                        f"pos_err={last_pos_err_norm: .3f}m "
                        f"vel_err={last_vel_err_norm: .3f}m/s "
                        f"thrust={float(stack_state.last_debug['thrust_cmd'][0]): .3f}N "
                        f"sat_rate_1s={sat_rate: .2%}"
                    )

            # Keep camera centered on the vehicle as it moves.
            viewer.cam.lookat[:] = data.xpos[body_id]

            est = EstimatedStateBatch(
                pos_world=data.qpos[0:3][None, :],
                vel_world=data.cvel[body_id, 3:6][None, :],
                quat_world_body=data.xquat[body_id][None, :],
                omega_body=data.sensordata[gyro_adr : gyro_adr + 3][None, :],
            )
            motor_out, stack_state, _ = stack.controller_step(
                state=stack_state,
                est=est,
            )
            sat_window.append(float(bool(motor_out.saturated[0])))
            for i, actuator_id in enumerate(actuator_ids):
                data.ctrl[actuator_id] = float(motor_out.motor[0, i])

            mujoco.mj_step(model, data)
            viewer.sync()
            inner_idx += 1

            if not args.no_realtime:
                sim_elapsed = data.time - sim_start
                wall_elapsed = time.perf_counter() - wall_start
                sleep_time = sim_elapsed - wall_elapsed
                if sleep_time > 0:
                    time.sleep(min(sleep_time, 0.002))

    if args.traj_diagnostics:
        if live_plotter is not None:
            live_plotter.finalize(diag_log)
        else:
            # Backward-compatible mode: render diagnostics only after viewer closes.
            LiveDiagnosticsPlotter(update_hz=10.0).finalize(diag_log)


if __name__ == "__main__":
    main()
