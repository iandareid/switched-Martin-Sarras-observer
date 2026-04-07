from pathlib import Path
import tempfile
import unittest
from unittest import mock

from observer_labeling.scripts.preview_and_label_trajectory import build_preview_command, run_preview_and_label


def _make_config() -> dict:
    return {
        "data": {
            "trajectory_path": "observer_labeling/data/seeded_human_like.npz",
            "seed": 9,
            "duration": 12.5,
            "traj_dt": 0.1,
        },
        "env": {
            "estimator_rate_hz": 400.0,
            "decision_rate_hz": 4.0,
            "hold_steps": 100,
        },
        "label_search": {
            "target_depth": 7,
            "w_attitude": 5.0,
            "w_gyro_bias": 30.0,
            "w_accel_bias": 30.0,
        },
    }


class PreviewAndLabelTests(unittest.TestCase):
    def test_build_preview_command_uses_data_config(self) -> None:
        config = _make_config()
        command = build_preview_command(config)
        self.assertEqual(command[0], mock.ANY)
        self.assertEqual(command[1], str(Path.cwd() / "run_quadrotor.py"))
        self.assertEqual(
            command[2:],
            ["--human-like", "--seed", "9", "--duration", "12.5", "--traj-dt", "0.1"],
        )

    def test_run_preview_and_label_runs_preview_then_pipeline(self) -> None:
        config = _make_config()
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "results"
            dataset_override = Path(tmpdir) / "dataset.npz"
            with (
                mock.patch("observer_labeling.scripts.preview_and_label_trajectory.subprocess.run") as preview_run,
                mock.patch(
                    "observer_labeling.scripts.preview_and_label_trajectory.run_pipeline",
                    return_value={"dataset_path": dataset_override, "generated_dataset": True, "output_dir": output_dir, "target_depth": 5, "backend": "cpu", "search_completion_rate": 1.0, "elapsed_sec": 1.2},
                ) as pipeline_run,
            ):
                result = run_preview_and_label(
                    config,
                    Path(tmpdir) / "labeling.yaml",
                    output_dir,
                    dataset_path_override=dataset_override,
                    target_depth_override=5,
                    traj_diagnostics=True,
                    no_realtime=True,
                )

            preview_run.assert_called_once()
            preview_args = preview_run.call_args.kwargs["args"] if "args" in preview_run.call_args.kwargs else preview_run.call_args.args[0]
            self.assertIn("--traj-diagnostics", preview_args)
            self.assertIn("--no-realtime", preview_args)
            pipeline_run.assert_called_once_with(
                config,
                Path(tmpdir) / "labeling.yaml",
                output_dir,
                dataset_path_override=dataset_override,
                target_depth_override=5,
                force_regenerate_dataset=True,
            )
            self.assertEqual(result["target_depth"], 5)

    def test_run_preview_and_label_can_skip_preview(self) -> None:
        config = _make_config()
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "results"
            with (
                mock.patch("observer_labeling.scripts.preview_and_label_trajectory.subprocess.run") as preview_run,
                mock.patch(
                    "observer_labeling.scripts.preview_and_label_trajectory.run_pipeline",
                    return_value={"dataset_path": output_dir / "dataset.npz", "generated_dataset": True, "output_dir": output_dir, "target_depth": 7, "backend": "cpu", "search_completion_rate": 1.0, "elapsed_sec": 1.2},
                ) as pipeline_run,
            ):
                run_preview_and_label(
                    config,
                    Path(tmpdir) / "labeling.yaml",
                    output_dir,
                    skip_preview=True,
                )

            preview_run.assert_not_called()
            pipeline_run.assert_called_once()


if __name__ == "__main__":
    unittest.main()
