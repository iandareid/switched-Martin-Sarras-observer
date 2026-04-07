from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from observer_labeling.data.dataset import TrajectoryDataset


@dataclass(frozen=True)
class ObservationNormStats:
    mean: np.ndarray
    std: np.ndarray


def _synthetic_mode_one_hot(num_samples: int) -> np.ndarray:
    mode_ids = np.arange(num_samples, dtype=np.int64) % 3
    return np.eye(3, dtype=np.float64)[mode_ids]


def compute_observation_norm(dataset: TrajectoryDataset, eps: float = 1e-6) -> ObservationNormStats:
    gyro_bias = dataset.true_gyro_bias if dataset.true_gyro_bias is not None else dataset.true_bias
    if gyro_bias is None:
        gyro_bias = np.zeros((dataset.num_samples, 3), dtype=np.float64)
    accel_bias = dataset.true_accel_bias
    if accel_bias is None:
        accel_bias = np.zeros((dataset.num_samples, 3), dtype=np.float64)
    elapsed_time = (dataset.t - dataset.t[0]).reshape((-1, 1))
    obs = np.concatenate(
        [
            dataset.cmd_rates,
            dataset.throttle,
            dataset.gyro,
            dataset.accel,
            dataset.true_quat,
            gyro_bias,
            accel_bias,
            _synthetic_mode_one_hot(dataset.num_samples),
            elapsed_time,
        ],
        axis=1,
    )
    mean = np.mean(obs, axis=0)
    std = np.maximum(np.std(obs, axis=0), eps)
    return ObservationNormStats(mean=mean, std=std)


def normalize_observation(obs: np.ndarray, stats: ObservationNormStats) -> np.ndarray:
    return (obs - stats.mean) / stats.std
