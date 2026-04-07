from __future__ import annotations

import argparse
import csv
from pathlib import Path

from .backend import jnp
from .math_utils import matrix_to_euler321, rotation_error_deg
from .observer import default_config, init_state, mode_to_one_hot, scan


def _load_rows(path: Path):
    with path.open("r", newline="") as handle:
        return list(csv.DictReader(handle))


def _col(rows, name):
    return jnp.asarray([float(row[name]) for row in rows])


def _vec(rows, prefix):
    return jnp.stack([_col(rows, f"{prefix}_x"), _col(rows, f"{prefix}_y"), _col(rows, f"{prefix}_z")], axis=-1)


def _truth_rotation(rows):
    roll = _col(rows, "truth_roll")
    pitch = _col(rows, "truth_pitch")
    yaw = _col(rows, "truth_yaw")
    cr = jnp.cos(roll)
    sr = jnp.sin(roll)
    cp = jnp.cos(pitch)
    sp = jnp.sin(pitch)
    cy = jnp.cos(yaw)
    sy = jnp.sin(yaw)
    rx = jnp.stack(
        [
            jnp.stack([jnp.ones_like(roll), jnp.zeros_like(roll), jnp.zeros_like(roll)], axis=-1),
            jnp.stack([jnp.zeros_like(roll), cr, -sr], axis=-1),
            jnp.stack([jnp.zeros_like(roll), sr, cr], axis=-1),
        ],
        axis=-2,
    )
    ry = jnp.stack(
        [
            jnp.stack([cp, jnp.zeros_like(roll), sp], axis=-1),
            jnp.stack([jnp.zeros_like(roll), jnp.ones_like(roll), jnp.zeros_like(roll)], axis=-1),
            jnp.stack([-sp, jnp.zeros_like(roll), cp], axis=-1),
        ],
        axis=-2,
    )
    rz = jnp.stack(
        [
            jnp.stack([cy, -sy, jnp.zeros_like(roll)], axis=-1),
            jnp.stack([sy, cy, jnp.zeros_like(roll)], axis=-1),
            jnp.stack([jnp.zeros_like(roll), jnp.zeros_like(roll), jnp.ones_like(roll)], axis=-1),
        ],
        axis=-2,
    )
    return jnp.matmul(rz, jnp.matmul(ry, rx))


def run_parity_demo(reference_csv: Path, output_csv: Path):
    rows = _load_rows(reference_csv)
    if len(rows) < 2:
        raise ValueError("reference CSV must contain at least two rows")

    t = _col(rows, "t")
    dt = float(t[1] - t[0])
    alpha_i = jnp.asarray([float(rows[0]["truth_alpha_x"]), float(rows[0]["truth_alpha_y"]), float(rows[0]["truth_alpha_z"])])
    beta_i = jnp.asarray([float(rows[0]["truth_beta_x"]), float(rows[0]["truth_beta_y"]), float(rows[0]["truth_beta_z"])])
    config = default_config(alpha_i, beta_i)

    obs_seq = {
        "omega_m": _vec(rows, "meas_omega") if "meas_omega_x" in rows[0] else _vec(rows, "truth_bg"),
        "alpha_m": _vec(rows, "meas_alpha"),
        "beta_m": _vec(rows, "meas_beta"),
    }
    mode_seq = mode_to_one_hot(jnp.asarray([int(row["mode"]) for row in rows]))

    state0 = init_state(
        alpha_hat0=alpha_i,
        beta_hat0=beta_i,
        b_hat0=jnp.zeros(3),
        b_alpha_hat0=jnp.zeros(3),
        mode0=int(rows[0]["mode"]),
        config=config,
    )

    _, outputs = scan(state0, obs_seq, mode_seq, dt, config)
    est_euler = matrix_to_euler321(outputs.r_hat)
    truth_r = _truth_rotation(rows)
    attitude_error_deg = rotation_error_deg(outputs.r_hat, truth_r)
    truth_bg = _vec(rows, "truth_bg")
    truth_ba = _vec(rows, "truth_ba")
    gyro_bias_error = jnp.linalg.norm(outputs.b_hat - truth_bg, axis=-1)
    accel_bias_error = jnp.linalg.norm(outputs.b_alpha_hat - truth_ba, axis=-1)
    alpha_corr = obs_seq["alpha_m"] - outputs.b_alpha_hat
    alpha_corr_n = alpha_corr / jnp.maximum(jnp.linalg.norm(alpha_corr, axis=-1, keepdims=True), 1e-12)
    g_hat_n = outputs.g_hat_sw / jnp.maximum(jnp.linalg.norm(outputs.g_hat_sw, axis=-1, keepdims=True), 1e-12)
    r_a_deg = jnp.degrees(jnp.arccos(jnp.clip(jnp.sum(alpha_corr_n * g_hat_n, axis=-1), -1.0, 1.0)))

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with output_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for i, row in enumerate(rows):
            out_row = dict(row)
            out_row["t"] = f"{float(t[i]):.9f}"
            out_row["mode"] = str(int(outputs.mode_index[i]))
            out_row["switch_flag"] = str(int(outputs.switch_flag[i]))
            out_row["est_roll"] = f"{float(est_euler[i, 0]):.9f}"
            out_row["est_pitch"] = f"{float(est_euler[i, 1]):.9f}"
            out_row["est_yaw"] = f"{float(est_euler[i, 2]):.9f}"
            out_row["est_bg_x"] = f"{float(outputs.b_hat[i, 0]):.9f}"
            out_row["est_bg_y"] = f"{float(outputs.b_hat[i, 1]):.9f}"
            out_row["est_bg_z"] = f"{float(outputs.b_hat[i, 2]):.9f}"
            out_row["est_ba_x"] = f"{float(outputs.b_alpha_hat[i, 0]):.9f}"
            out_row["est_ba_y"] = f"{float(outputs.b_alpha_hat[i, 1]):.9f}"
            out_row["est_ba_z"] = f"{float(outputs.b_alpha_hat[i, 2]):.9f}"
            out_row["g_hat_sw_x"] = f"{float(outputs.g_hat_sw[i, 0]):.9f}"
            out_row["g_hat_sw_y"] = f"{float(outputs.g_hat_sw[i, 1]):.9f}"
            out_row["g_hat_sw_z"] = f"{float(outputs.g_hat_sw[i, 2]):.9f}"
            out_row["m_hat_sw_x"] = f"{float(outputs.m_hat_sw[i, 0]):.9f}"
            out_row["m_hat_sw_y"] = f"{float(outputs.m_hat_sw[i, 1]):.9f}"
            out_row["m_hat_sw_z"] = f"{float(outputs.m_hat_sw[i, 2]):.9f}"
            out_row["r_a_deg"] = f"{float(r_a_deg[i]):.9f}"
            out_row["attitude_error_deg"] = f"{float(attitude_error_deg[i]):.9f}"
            out_row["gyro_bias_error"] = f"{float(gyro_bias_error[i]):.9f}"
            out_row["accel_bias_error"] = f"{float(accel_bias_error[i]):.9f}"
            writer.writerow(out_row)


def main():
    parser = argparse.ArgumentParser(description="Replay the C++ observer CSV through the JAX switched observer.")
    parser.add_argument(
        "--reference-csv",
        type=Path,
        default=Path("results/three_mode/switched_demo_three_mode.csv"),
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("results/jax_three_mode/switched_demo_three_mode_jax.csv"),
    )
    args = parser.parse_args()
    run_parity_demo(args.reference_csv, args.output_csv)
    print(f"Wrote {args.output_csv}")


if __name__ == "__main__":
    main()
