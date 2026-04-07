import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

from observer_labeling.data.dataset import TrajectoryDataset, save_dataset
from observer_labeling.scripts.run_labeling_pipeline import ensure_dataset, run_pipeline


def _make_dataset(num_samples: int = 40) -> TrajectoryDataset:
    t = np.arange(num_samples, dtype=np.float64) / 400.0
    accel = np.tile(np.array([[0.0, 0.0, -1.0]], dtype=np.float64), (num_samples, 1))
    mag = np.tile(np.array([[1.0, 0.0, 0.0]], dtype=np.float64), (num_samples, 1))
    return TrajectoryDataset(
        t=t,
        cmd_rates=np.zeros((num_samples, 3), dtype=np.float64),
        throttle=np.zeros((num_samples, 1), dtype=np.float64),
        gyro=np.zeros((num_samples, 3), dtype=np.float64),
        accel=accel,
        mag_body=mag,
        true_quat=np.tile(np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float64), (num_samples, 1)),
        true_bias=np.zeros((num_samples, 3), dtype=np.float64),
        true_gyro_bias=np.zeros((num_samples, 3), dtype=np.float64),
        true_accel_bias=np.zeros((num_samples, 3), dtype=np.float64),
        trajectory_name="test_traj",
        estimator_rate_hz=400.0,
        decision_rate_hz=4.0,
        hold_steps=10,
        seed=7,
    )


def _make_config(dataset_path: Path) -> dict:
    return {
        "data": {
            "trajectory_path": str(dataset_path),
            "seed": 7,
            "duration": 2.0,
            "traj_dt": 0.05,
        },
        "env": {
            "estimator_rate_hz": 400.0,
            "decision_rate_hz": 4.0,
            "hold_steps": 10,
        },
        "label_search": {
            "target_depth": 2,
            "w_attitude": 1.0,
            "w_gyro_bias": 1.0,
            "w_accel_bias": 1.0,
            "attitude_pre_break_gain": 1.0,
            "attitude_break_norm": 0.07,
            "attitude_transition_width": 0.01,
            "attitude_post_break_gain": 2.0,
            "gyro_bias_pre_break_gain": 1.0,
            "gyro_bias_break_norm": 0.04,
            "gyro_bias_transition_width": 0.01,
            "gyro_bias_post_break_gain": 2.0,
            "accel_bias_pre_break_gain": 1.0,
            "accel_bias_break_norm": 0.04,
            "accel_bias_transition_width": 0.01,
            "accel_bias_post_break_gain": 2.0,
            "mahony": {
                "k_p": 2.0,
                "k_i": 0.1,
                "accel_weight": 1.0,
                "mag_weight": 1.0,
                "accel_gate_margin_mps2": 0.5,
                "g_ref_mps2": 9.81,
            },
        },
    }


class LabelingPipelineTests(unittest.TestCase):
    def test_ensure_dataset_generates_missing_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_path = Path(tmpdir) / "generated.npz"
            config = _make_config(dataset_path)
            with mock.patch(
                "observer_labeling.scripts.run_labeling_pipeline.record_human_like_dataset",
                return_value=_make_dataset(),
            ) as recorder:
                resolved_path, generated = ensure_dataset(config, dataset_path, force_regenerate=False)

            self.assertTrue(generated)
            self.assertEqual(resolved_path, dataset_path)
            self.assertTrue(dataset_path.exists())
            recorder.assert_called_once()

    def test_run_pipeline_writes_expected_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            dataset_path = tmpdir_path / "seeded.npz"
            output_dir = tmpdir_path / "results"
            save_dataset(dataset_path, _make_dataset())
            config = _make_config(dataset_path)

            result = run_pipeline(
                config,
                tmpdir_path / "labeling.yaml",
                output_dir,
                target_depth_override=2,
            )

            self.assertEqual(result["target_depth"], 2)
            self.assertTrue((output_dir / "summary.csv").exists())
            self.assertTrue((output_dir / "decision_records.csv").exists())
            self.assertTrue((output_dir / "run_metadata.json").exists())
            self.assertTrue((output_dir / "labeled_trajectory_depth2.npz").exists())

            metadata = json.loads((output_dir / "run_metadata.json").read_text(encoding="utf-8"))
            self.assertEqual(metadata["target_depth"], 2)
            self.assertEqual(metadata["dataset_path"], str(dataset_path))
            self.assertIn("attitude_error", metadata["plot_files"])


if __name__ == "__main__":
    unittest.main()
