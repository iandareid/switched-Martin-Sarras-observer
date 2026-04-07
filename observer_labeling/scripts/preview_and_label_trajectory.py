from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from observer_labeling.scripts.run_labeling_pipeline import run_pipeline


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
        "--traj-diagnostics",
        action="store_true",
        help="Print generated trajectory segment diagnostics during the MuJoCo preview.",
    )
    parser.add_argument(
        "--no-realtime",
        action="store_true",
        help="Run the MuJoCo preview as fast as possible.",
    )
    parser.add_argument(
        "--skip-preview",
        action="store_true",
        help="Skip the MuJoCo preview and run labeling directly.",
    )
    return parser.parse_args()


def _load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def build_preview_command(config: dict) -> list[str]:
    data_cfg = config["data"]
    return [
        sys.executable,
        str(REPO_ROOT / "run_quadrotor.py"),
        "--human-like",
        "--seed",
        str(int(data_cfg.get("seed", 7))),
        "--duration",
        str(float(data_cfg.get("duration", 25.0))),
        "--traj-dt",
        str(float(data_cfg.get("traj_dt", 0.05))),
    ]


def run_preview_and_label(
    config: dict,
    config_path: Path,
    output_dir: Path,
    *,
    dataset_path_override: Path | None = None,
    target_depth_override: int | None = None,
    traj_diagnostics: bool = False,
    no_realtime: bool = False,
    skip_preview: bool = False,
) -> dict[str, object]:
    if not skip_preview:
        preview_cmd = build_preview_command(config)
        if traj_diagnostics:
            preview_cmd.append("--traj-diagnostics")
        if no_realtime:
            preview_cmd.append("--no-realtime")
        subprocess.run(preview_cmd, check=True, cwd=REPO_ROOT)

    return run_pipeline(
        config,
        config_path,
        output_dir,
        dataset_path_override=dataset_path_override,
        target_depth_override=target_depth_override,
        force_regenerate_dataset=True,
    )


def main() -> None:
    args = parse_args()
    config = _load_config(args.config)
    result = run_preview_and_label(
        config,
        args.config,
        args.output_dir,
        dataset_path_override=args.dataset_path,
        target_depth_override=args.target_depth,
        traj_diagnostics=args.traj_diagnostics,
        no_realtime=args.no_realtime,
        skip_preview=args.skip_preview,
    )
    print(
        f"dataset={result['dataset_path']} generated={result['generated_dataset']} "
        f"output_dir={result['output_dir']} depth={result['target_depth']} "
        f"backend={result['backend']} completion_rate={result['search_completion_rate']:.3f} "
        f"elapsed_sec={result['elapsed_sec']:.3f}"
    )


if __name__ == "__main__":
    main()
