from __future__ import annotations

from flax import struct
import jax
import jax.numpy as jnp
import numpy as np

from observer_labeling.data.dataset import TrajectoryDataset


@struct.dataclass
class JaxTrajectoryDataset:
    t: jnp.ndarray
    cmd_rates: jnp.ndarray
    throttle: jnp.ndarray
    gyro: jnp.ndarray
    accel: jnp.ndarray
    mag_body: jnp.ndarray
    true_quat: jnp.ndarray
    true_gyro_bias: jnp.ndarray
    true_accel_bias: jnp.ndarray
    first_valid_idx: jnp.ndarray

    @property
    def num_samples(self) -> int:
        return int(self.t.shape[0])


@struct.dataclass
class JaxObservationNormStats:
    mean: jnp.ndarray
    std: jnp.ndarray


def _synthetic_mode_one_hot_jax(num_samples: int) -> jnp.ndarray:
    mode_ids = jnp.arange(num_samples, dtype=jnp.int32) % 3
    return jax.nn.one_hot(mode_ids, 3, dtype=jnp.float32)


def to_jax_dataset(dataset: TrajectoryDataset) -> JaxTrajectoryDataset:
    if dataset.mag_body is None:
        raise ValueError("Trajectory dataset must include mag_body for JAX training.")
    true_gyro_bias = dataset.true_gyro_bias if dataset.true_gyro_bias is not None else dataset.true_bias
    if true_gyro_bias is None:
        true_gyro_bias = jnp.zeros_like(jnp.asarray(dataset.gyro, dtype=jnp.float32))
    true_accel_bias = dataset.true_accel_bias
    if true_accel_bias is None:
        true_accel_bias = jnp.zeros_like(jnp.asarray(dataset.accel, dtype=jnp.float32))
    quat_norm = np.linalg.norm(dataset.true_quat, axis=1)
    accel_norm = np.linalg.norm(dataset.accel, axis=1)
    valid = np.isfinite(quat_norm) & (quat_norm > 0.5) & np.isfinite(accel_norm) & (accel_norm > 1e-3)
    if not np.any(valid):
        raise ValueError("Trajectory dataset does not contain any valid initialization sample.")
    first_valid_idx = int(np.argmax(valid))
    return JaxTrajectoryDataset(
        t=jnp.asarray(dataset.t, dtype=jnp.float32),
        cmd_rates=jnp.asarray(dataset.cmd_rates, dtype=jnp.float32),
        throttle=jnp.asarray(dataset.throttle, dtype=jnp.float32),
        gyro=jnp.asarray(dataset.gyro, dtype=jnp.float32),
        accel=jnp.asarray(dataset.accel, dtype=jnp.float32),
        mag_body=jnp.asarray(dataset.mag_body, dtype=jnp.float32),
        true_quat=jnp.asarray(dataset.true_quat, dtype=jnp.float32),
        true_gyro_bias=jnp.asarray(true_gyro_bias, dtype=jnp.float32),
        true_accel_bias=jnp.asarray(true_accel_bias, dtype=jnp.float32),
        first_valid_idx=jnp.asarray(first_valid_idx, dtype=jnp.int32),
    )


def compute_observation_norm_jax(dataset: JaxTrajectoryDataset, eps: float = 1e-6) -> JaxObservationNormStats:
    elapsed_time = (dataset.t - dataset.t[0]).reshape((-1, 1))
    obs = jnp.concatenate(
        [
            dataset.cmd_rates,
            dataset.throttle,
            dataset.gyro,
            dataset.accel,
            dataset.true_quat,
            dataset.true_gyro_bias,
            dataset.true_accel_bias,
            _synthetic_mode_one_hot_jax(dataset.num_samples),
            elapsed_time,
        ],
        axis=1,
    )
    mean = jnp.mean(obs, axis=0)
    std = jnp.maximum(jnp.std(obs, axis=0), jnp.asarray(eps, dtype=jnp.float32))
    return JaxObservationNormStats(mean=mean, std=std)


def normalize_observation_jax(obs: jnp.ndarray, stats: JaxObservationNormStats) -> jnp.ndarray:
    return (obs - stats.mean) / stats.std
