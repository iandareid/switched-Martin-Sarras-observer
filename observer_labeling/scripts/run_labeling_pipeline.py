from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
import time

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from observer_labeling.data.dataset import load_dataset, save_dataset
from observer_labeling.data.jax_dataset import to_jax_dataset
from observer_labeling.data.recording import RecorderConfig, record_human_like_dataset
from observer_labeling.eval.label_search import (
    MODE_LABELS,
    build_mahony_params_from_config,
    build_search_problem_from_config,
    label_trajectory,
    save_labeled_trajectory,
)
from observer_labeling.eval.plots import save_labeled_trajectory_plot_sets
from observer_labeling.runtime import GpuUnavailableError, configure_jax_gpu, detect_jax_backend


configure_jax_gpu()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("observer_labeling/configs/labeling.yaml"),
        help="Labeling pipeline YAML config.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("observer_labeling/results/labeling_run"),
        help="Directory for artifacts, plots, and summaries.",
    )
    parser.add_argument("--target-depth", type=int, default=None, help="Optional search depth override.")
    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=None,
        help="Optional dataset path override. Defaults to data.trajectory_path in the config.",
    )
    parser.add_argument(
        "--force-regenerate-dataset",
        action="store_true",
        help="Regenerate the recorded trajectory even if the dataset already exists.",
    )
    return parser.parse_args()


def _load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _dataset_path(config: dict, override: Path | None) -> Path:
    if override is not None:
        return override
    return Path(config["data"]["trajectory_path"])


def ensure_dataset(config: dict, dataset_path: Path, force_regenerate: bool) -> tuple[Path, bool]:
    if dataset_path.exists() and not force_regenerate:
        return dataset_path, False
    dataset = record_human_like_dataset(
        RecorderConfig(
            seed=int(config["data"].get("seed", 7)),
            duration=float(config["data"].get("duration", 25.0)),
            traj_dt=float(config["data"].get("traj_dt", 0.05)),
        )
    )
    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    save_dataset(dataset_path, dataset)
    return dataset_path, True


def run_pipeline(
    config: dict,
    config_path: Path,
    output_dir: Path,
    *,
    dataset_path_override: Path | None = None,
    target_depth_override: int | None = None,
    force_regenerate_dataset: bool = False,
) -> dict[str, object]:
    try:
        backend, device_kinds = detect_jax_backend()
    except GpuUnavailableError as exc:
        raise SystemExit(str(exc)) from exc

    dataset_path = _dataset_path(config, dataset_path_override)
    dataset_path, generated_dataset = ensure_dataset(config, dataset_path, force_regenerate_dataset)
    dataset = load_dataset(dataset_path)
    jax_dataset = to_jax_dataset(dataset)
    problem, _ = build_search_problem_from_config(config, jax_dataset)

    target_depth = int(target_depth_override) if target_depth_override is not None else int(config["label_search"].get("target_depth", 7))

    output_dir.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    last_progress_percent = -1

    def report_progress(fraction_complete: float) -> None:
        nonlocal last_progress_percent
        percent_complete = int(np.clip(np.floor(100.0 * fraction_complete), 0.0, 100.0))
        if percent_complete != last_progress_percent:
            print(f"trajectory_progress={percent_complete}%", flush=True)
            last_progress_percent = percent_complete

    trace = label_trajectory(
        problem,
        depth=target_depth,
        mahony_params=build_mahony_params_from_config(config, problem.dataset),
        progress_callback=report_progress,
    )
    elapsed_sec = time.perf_counter() - start

    artifact_path = output_dir / f"labeled_trajectory_depth{target_depth}.npz"
    save_labeled_trajectory(trace, artifact_path)
    plot_paths = save_labeled_trajectory_plot_sets(trace, output_dir)

    summary_rows = [
        {
            "depth": target_depth,
            "num_samples": int(trace.t.shape[0]),
            "num_decisions": int(trace.actions.shape[0]),
            "num_switches": int(trace.switch_actions.shape[0]),
            "mean_attitude_error": float(np.mean(trace.attitude_error)),
            "max_attitude_error": float(np.max(trace.attitude_error)),
            "mahony_mean_attitude_error": float(np.mean(trace.mahony_attitude_error)),
            "mahony_max_attitude_error": float(np.max(trace.mahony_attitude_error)),
            "mean_gyro_bias_error_norm": float(np.mean(trace.bias_error_norm)),
            "max_gyro_bias_error_norm": float(np.max(trace.bias_error_norm)),
            "mahony_mean_gyro_bias_error_norm": float(np.mean(trace.mahony_bias_error_norm)),
            "mahony_max_gyro_bias_error_norm": float(np.max(trace.mahony_bias_error_norm)),
            "mean_accel_bias_error_norm": float(np.mean(trace.accel_bias_error_norm)),
            "max_accel_bias_error_norm": float(np.max(trace.accel_bias_error_norm)),
            "search_completion_rate": float(np.mean(trace.search_completed)),
            "elapsed_sec": elapsed_sec,
        }
    ]
    _write_csv(output_dir / "summary.csv", summary_rows)

    decision_rows = [
        {
            "decision_index": idx,
            "sample_idx": int(trace.decision_sample_idx[idx]),
            "time_sec": float(trace.decision_t[idx]),
            "best_action": int(trace.actions[idx]),
            "best_label": MODE_LABELS[int(trace.actions[idx])],
            "cost_go": float(trace.root_costs[idx, 0]),
            "cost_bvo": float(trace.root_costs[idx, 1]),
            "cost_pae": float(trace.root_costs[idx, 2]),
            "completed": int(trace.search_completed[idx]),
        }
        for idx in range(trace.actions.shape[0])
    ]
    _write_csv(output_dir / "decision_records.csv", decision_rows)

    metadata = {
        "mode": "labeling_pipeline",
        "config_path": str(config_path),
        "dataset_path": str(dataset_path),
        "generated_dataset": generated_dataset,
        "jax_backend": backend,
        "device_kinds": device_kinds,
        "target_depth": target_depth,
        "label_artifact": str(artifact_path),
        "elapsed_sec": elapsed_sec,
        "env": dict(config["env"]),
        "label_search": dict(config["label_search"]),
        "plot_files": {key: str(path) for key, path in plot_paths.items()},
    }
    with (output_dir / "run_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    return {
        "dataset_path": dataset_path,
        "generated_dataset": generated_dataset,
        "output_dir": output_dir,
        "target_depth": target_depth,
        "backend": backend,
        "elapsed_sec": elapsed_sec,
        "artifact_path": artifact_path,
        "search_completion_rate": float(np.mean(trace.search_completed)),
    }


def main() -> None:
    args = parse_args()
    config = _load_config(args.config)
    result = run_pipeline(
        config,
        args.config,
        args.output_dir,
        dataset_path_override=args.dataset_path,
        target_depth_override=args.target_depth,
        force_regenerate_dataset=args.force_regenerate_dataset,
    )
    print(
        f"dataset={result['dataset_path']} generated={result['generated_dataset']} "
        f"output_dir={result['output_dir']} depth={result['target_depth']} "
        f"backend={result['backend']} completion_rate={result['search_completion_rate']:.3f} "
        f"elapsed_sec={result['elapsed_sec']:.3f}"
    )


if __name__ == "__main__":
    main()
