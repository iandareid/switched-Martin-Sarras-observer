from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from observer_labeling.eval.label_search import LabeledTrajectoryTrace


MODE_LABELS = {
    0: "GO",
    1: "BVO",
    2: "PAE",
}


def _quat_to_euler_zyx(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float64)
    norm = np.linalg.norm(quat, axis=1, keepdims=True)
    norm = np.where(norm > 0.0, norm, 1.0)
    quat = quat / norm
    w = quat[:, 0]
    x = quat[:, 1]
    y = quat[:, 2]
    z = quat[:, 3]

    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    sinp = 2.0 * (w * y - z * x)
    pitch = np.arcsin(np.clip(sinp, -1.0, 1.0))

    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)

    return np.stack((roll, pitch, yaw), axis=1)


def _draw_switch_lines(ax, switch_times: np.ndarray, switch_actions: np.ndarray) -> None:
    ymin, ymax = ax.get_ylim()
    ytext = ymax - 0.04 * (ymax - ymin if ymax > ymin else 1.0)
    for t, action in zip(switch_times, switch_actions, strict=False):
        ax.axvline(float(t), color="crimson", linestyle="--", linewidth=1.0, alpha=0.8)
        label = MODE_LABELS.get(int(action), f"mode={int(action)}")
        ax.text(
            float(t),
            ytext,
            label,
            rotation=90,
            color="crimson",
            fontsize=8,
            va="top",
            ha="right",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.7, "pad": 1.0},
        )

def save_labeled_trajectory_plots(
    trace: LabeledTrajectoryTrace,
    output_dir: str | Path,
    include_mahony: bool = True,
    filename_suffix: str = "",
) -> dict[str, Path]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    t = np.asarray(trace.t)
    true_quat = np.asarray(trace.true_quat)
    est_quat = np.asarray(trace.est_quat)
    mahony_quat = np.asarray(trace.mahony_quat)
    attitude_error = np.asarray(trace.attitude_error)
    mahony_attitude_error = np.asarray(trace.mahony_attitude_error)
    bias_error = np.asarray(trace.bias_error)
    bias_error_norm = np.asarray(trace.bias_error_norm)
    mahony_bias_error = np.asarray(trace.mahony_bias_error)
    mahony_bias_error_norm = np.asarray(trace.mahony_bias_error_norm)
    accel_bias_error = np.asarray(trace.accel_bias_error)
    accel_bias_error_norm = np.asarray(trace.accel_bias_error_norm)
    decision_t = np.asarray(trace.decision_t)
    root_costs = np.asarray(trace.root_costs)
    switch_times = np.asarray(trace.switch_times)
    switch_actions = np.asarray(trace.switch_actions)

    attitude_path = output_path / f"attitude_error_over_time{filename_suffix}.png"
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(t, attitude_error, color="navy", linewidth=2.0, label="Switched Observer")
    if include_mahony:
        ax.plot(t, mahony_attitude_error, color="#ff7f0e", linewidth=1.8, linestyle="--", label="Standalone Mahony")
    ax.set_title("Attitude Error Over Time")
    _draw_switch_lines(ax, switch_times, switch_actions)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Quaternion Angle Error [rad]")
    ax.grid(True, alpha=0.3)
    if include_mahony:
        ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(attitude_path, dpi=160)
    plt.close(fig)

    bias_components_path = output_path / f"gyro_bias_error_components_over_time{filename_suffix}.png"
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    labels = ("x", "y", "z")
    colors = ("#1f77b4", "#ff7f0e", "#2ca02c")
    for idx, ax in enumerate(axes):
        ax.plot(t, bias_error[:, idx], color=colors[idx], linewidth=1.8, label="Switched Observer")
        if include_mahony:
            ax.plot(t, mahony_bias_error[:, idx], color=colors[idx], linewidth=1.4, linestyle="--", label="Standalone Mahony")
        ax.set_ylabel(f"{labels[idx]} err")
        ax.grid(True, alpha=0.3)
        _draw_switch_lines(ax, switch_times, switch_actions)
    axes[0].set_title("Gyro Bias Error Components Over Time")
    if include_mahony:
        axes[0].legend(loc="best")
    axes[-1].set_xlabel("Time [s]")
    fig.tight_layout()
    fig.savefig(bias_components_path, dpi=160)
    plt.close(fig)

    bias_norm_path = output_path / f"gyro_bias_error_norm_over_time{filename_suffix}.png"
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(t, bias_error_norm, color="darkgreen", linewidth=2.0, label="Switched Observer")
    if include_mahony:
        ax.plot(t, mahony_bias_error_norm, color="#ff9896", linewidth=1.8, linestyle="--", label="Standalone Mahony")
    ax.set_title("Gyro Bias Error Norm Over Time")
    _draw_switch_lines(ax, switch_times, switch_actions)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Bias Error Norm")
    ax.grid(True, alpha=0.3)
    if include_mahony:
        ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(bias_norm_path, dpi=160)
    plt.close(fig)

    accel_bias_components_path = output_path / f"accel_bias_error_components_over_time{filename_suffix}.png"
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    accel_colors = ("#bcbd22", "#7f7f7f", "#e377c2")
    for idx, ax in enumerate(axes):
        ax.plot(t, accel_bias_error[:, idx], color=accel_colors[idx], linewidth=1.8, label="Switched Observer")
        ax.set_ylabel(f"{labels[idx]} err")
        ax.grid(True, alpha=0.3)
        _draw_switch_lines(ax, switch_times, switch_actions)
    axes[0].set_title("Accel Bias Error Components Over Time")
    if include_mahony:
        axes[0].legend(loc="best")
    axes[-1].set_xlabel("Time [s]")
    fig.tight_layout()
    fig.savefig(accel_bias_components_path, dpi=160)
    plt.close(fig)

    accel_bias_norm_path = output_path / f"accel_bias_error_norm_over_time{filename_suffix}.png"
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(t, accel_bias_error_norm, color="#8c564b", linewidth=2.0, label="Switched Observer")
    ax.set_title("Accel Bias Error Norm Over Time")
    _draw_switch_lines(ax, switch_times, switch_actions)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Accel Bias Error Norm")
    ax.grid(True, alpha=0.3)
    if include_mahony:
        ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(accel_bias_norm_path, dpi=160)
    plt.close(fig)

    attitude_compare_path = output_path / f"attitude_truth_vs_estimate_over_time{filename_suffix}.png"
    true_euler = _quat_to_euler_zyx(true_quat)
    est_euler = _quat_to_euler_zyx(est_quat)
    mahony_euler = _quat_to_euler_zyx(mahony_quat)
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    labels = ("Roll", "Pitch", "Yaw")
    colors = ("#8c564b", "#9467bd", "#17becf")
    for idx, ax in enumerate(axes):
        ax.plot(t, true_euler[:, idx], color=colors[idx], linewidth=2.0, label="Truth")
        ax.plot(t, est_euler[:, idx], color=colors[idx], linewidth=1.6, linestyle="-", label="Switched Observer")
        if include_mahony:
            ax.plot(t, mahony_euler[:, idx], color=colors[idx], linewidth=1.4, linestyle="--", label="Standalone Mahony")
        ax.set_ylabel(f"{labels[idx]} [rad]")
        ax.grid(True, alpha=0.3)
        _draw_switch_lines(ax, switch_times, switch_actions)
    axes[0].set_title("Truth vs Estimated Attitude Over Time")
    axes[0].legend(loc="best")
    axes[-1].set_xlabel("Time [s]")
    fig.tight_layout()
    fig.savefig(attitude_compare_path, dpi=160)
    plt.close(fig)

    search_cost_path = output_path / f"label_search_cost_over_time{filename_suffix}.png"
    fig, ax = plt.subplots(figsize=(10, 4))
    cost_labels = ("GO", "BVO", "PAE")
    cost_colors = ("#d62728", "#1f77b4", "#2ca02c")
    for idx, label in enumerate(cost_labels):
        ax.plot(decision_t, root_costs[:, idx], label=label, color=cost_colors[idx], linewidth=1.8)
    ax.set_title("Label Search Root Cost Over Time")
    _draw_switch_lines(ax, switch_times, switch_actions)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Depth-Limited Cost")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(search_cost_path, dpi=160)
    plt.close(fig)

    return {
        "attitude_error": attitude_path,
        "bias_error_components": bias_components_path,
        "bias_error_norm": bias_norm_path,
        "accel_bias_error_components": accel_bias_components_path,
        "accel_bias_error_norm": accel_bias_norm_path,
        "attitude_truth_vs_estimate": attitude_compare_path,
        "label_search_cost": search_cost_path,
    }


def save_labeled_trajectory_plot_sets(trace: LabeledTrajectoryTrace, output_dir: str | Path) -> dict[str, Path]:
    comparison_paths = save_labeled_trajectory_plots(trace, output_dir, include_mahony=True, filename_suffix="")
    solo_paths = save_labeled_trajectory_plots(trace, output_dir, include_mahony=False, filename_suffix="_label_only")
    return {
        **comparison_paths,
        **{f"{key}_label_only": path for key, path in solo_paths.items()},
    }
