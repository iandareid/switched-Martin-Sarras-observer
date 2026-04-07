from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

OBS_DIM = 24


@dataclass(frozen=True)
class TrajectoryDataset:
    """Offline recorded trajectory dataset for observer labeling."""

    t: np.ndarray
    cmd_rates: np.ndarray
    throttle: np.ndarray
    gyro: np.ndarray
    accel: np.ndarray
    true_quat: np.ndarray
    mag_body: np.ndarray | None = None
    true_bias: np.ndarray | None = None
    true_gyro_bias: np.ndarray | None = None
    true_accel_bias: np.ndarray | None = None
    trajectory_name: str = ""
    cmd_rates_source: str = ""
    estimator_rate_hz: float = 400.0
    decision_rate_hz: float = 20.0
    hold_steps: int = 20
    seed: int = 0

    def validate(self) -> None:
        n = int(self.t.shape[0])
        if self.t.ndim != 1:
            raise ValueError("t must have shape [T].")
        if self.cmd_rates.shape != (n, 3):
            raise ValueError("cmd_rates must have shape [T, 3].")
        if self.throttle.shape != (n, 1):
            raise ValueError("throttle must have shape [T, 1].")
        if self.gyro.shape != (n, 3):
            raise ValueError("gyro must have shape [T, 3].")
        if self.accel.shape != (n, 3):
            raise ValueError("accel must have shape [T, 3].")
        if self.mag_body is not None and self.mag_body.shape != (n, 3):
            raise ValueError("mag_body must have shape [T, 3].")
        if self.true_quat.shape != (n, 4):
            raise ValueError("true_quat must have shape [T, 4].")
        if self.true_bias is not None and self.true_bias.shape[0] != n:
            raise ValueError("true_bias must have shape [T, B].")
        if self.true_gyro_bias is not None and self.true_gyro_bias.shape != (n, 3):
            raise ValueError("true_gyro_bias must have shape [T, 3].")
        if self.true_accel_bias is not None and self.true_accel_bias.shape != (n, 3):
            raise ValueError("true_accel_bias must have shape [T, 3].")

    @property
    def obs_dim(self) -> int:
        return OBS_DIM

    @property
    def num_samples(self) -> int:
        return int(self.t.shape[0])

    def slice_from(self, start_idx: int) -> "TrajectoryDataset":
        start = max(int(start_idx), 0)
        return TrajectoryDataset(
            t=self.t[start:],
            cmd_rates=self.cmd_rates[start:],
            throttle=self.throttle[start:],
            gyro=self.gyro[start:],
            accel=self.accel[start:],
            true_quat=self.true_quat[start:],
            mag_body=None if self.mag_body is None else self.mag_body[start:],
            true_bias=None if self.true_bias is None else self.true_bias[start:],
            true_gyro_bias=None if self.true_gyro_bias is None else self.true_gyro_bias[start:],
            true_accel_bias=None if self.true_accel_bias is None else self.true_accel_bias[start:],
            trajectory_name=self.trajectory_name,
            cmd_rates_source=self.cmd_rates_source,
            estimator_rate_hz=self.estimator_rate_hz,
            decision_rate_hz=self.decision_rate_hz,
            hold_steps=self.hold_steps,
            seed=self.seed,
        )

    def to_dict(self) -> dict[str, Any]:
        arrays: dict[str, Any] = {
            "t": self.t,
            "cmd_rates": self.cmd_rates,
            "throttle": self.throttle,
            "gyro": self.gyro,
            "accel": self.accel,
            "true_quat": self.true_quat,
            "trajectory_name": np.array(self.trajectory_name),
            "cmd_rates_source": np.array(self.cmd_rates_source),
            "estimator_rate_hz": np.array(self.estimator_rate_hz),
            "decision_rate_hz": np.array(self.decision_rate_hz),
            "hold_steps": np.array(self.hold_steps, dtype=np.int64),
            "seed": np.array(self.seed, dtype=np.int64),
        }
        if self.mag_body is not None:
            arrays["mag_body"] = self.mag_body
        if self.true_bias is not None:
            arrays["true_bias"] = self.true_bias
        if self.true_gyro_bias is not None:
            arrays["true_gyro_bias"] = self.true_gyro_bias
        if self.true_accel_bias is not None:
            arrays["true_accel_bias"] = self.true_accel_bias
        return arrays


def save_dataset(path: str | Path, dataset: TrajectoryDataset) -> None:
    dataset.validate()
    np.savez_compressed(Path(path), **dataset.to_dict())


def _first_valid_sample_idx(dataset: TrajectoryDataset) -> int:
    quat_norm = np.linalg.norm(dataset.true_quat, axis=1)
    accel_norm = np.linalg.norm(dataset.accel, axis=1)
    quat_valid = np.isfinite(quat_norm) & (quat_norm > 0.5)
    accel_valid = np.isfinite(accel_norm) & (accel_norm > 1e-3)
    valid = quat_valid & accel_valid
    if not np.any(valid):
        raise ValueError("Dataset does not contain any valid initial samples.")
    return int(np.argmax(valid))


def sanitize_dataset(dataset: TrajectoryDataset) -> TrajectoryDataset:
    start_idx = _first_valid_sample_idx(dataset)
    if start_idx == 0:
        return dataset
    trimmed = dataset.slice_from(start_idx)
    trimmed.validate()
    return trimmed


def load_dataset(path: str | Path) -> TrajectoryDataset:
    npz = np.load(Path(path), allow_pickle=False)
    mag_body = npz["mag_body"] if "mag_body" in npz.files else None
    true_bias = npz["true_bias"] if "true_bias" in npz.files else None
    true_gyro_bias = npz["true_gyro_bias"] if "true_gyro_bias" in npz.files else true_bias
    true_accel_bias = npz["true_accel_bias"] if "true_accel_bias" in npz.files else None
    dataset = TrajectoryDataset(
        t=np.asarray(npz["t"], dtype=np.float64),
        cmd_rates=np.asarray(npz["cmd_rates"], dtype=np.float64),
        throttle=np.asarray(npz["throttle"], dtype=np.float64),
        gyro=np.asarray(npz["gyro"], dtype=np.float64),
        accel=np.asarray(npz["accel"], dtype=np.float64),
        mag_body=None if mag_body is None else np.asarray(mag_body, dtype=np.float64),
        true_quat=np.asarray(npz["true_quat"], dtype=np.float64),
        true_bias=None if true_bias is None else np.asarray(true_bias, dtype=np.float64),
        true_gyro_bias=None if true_gyro_bias is None else np.asarray(true_gyro_bias, dtype=np.float64),
        true_accel_bias=None if true_accel_bias is None else np.asarray(true_accel_bias, dtype=np.float64),
        trajectory_name=str(npz["trajectory_name"].item()) if "trajectory_name" in npz.files else "",
        cmd_rates_source=str(npz["cmd_rates_source"].item()) if "cmd_rates_source" in npz.files else "",
        estimator_rate_hz=float(npz["estimator_rate_hz"]) if "estimator_rate_hz" in npz.files else 400.0,
        decision_rate_hz=float(npz["decision_rate_hz"]) if "decision_rate_hz" in npz.files else 20.0,
        hold_steps=int(npz["hold_steps"]) if "hold_steps" in npz.files else 20,
        seed=int(npz["seed"]) if "seed" in npz.files else 0,
    )
    dataset.validate()
    return sanitize_dataset(dataset)
