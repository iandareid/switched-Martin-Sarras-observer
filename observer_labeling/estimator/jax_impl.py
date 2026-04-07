from __future__ import annotations

from dataclasses import dataclass
from functools import partial

from flax import struct
import jax
import jax.numpy as jnp
import numpy as np

from jax_switched_observer.observer import default_config, init_state, mode_to_one_hot, step as observer_step
from observer_labeling.data.dataset import TrajectoryDataset
from observer_labeling.estimator.interface import EstimatorState


@struct.dataclass
class JaxEstimatorRuntimeState:
    attitude_quat: jnp.ndarray
    bias: jnp.ndarray
    accel_bias: jnp.ndarray
    observer_state: object


def _normalize(vec: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    if norm <= 1e-9:
        return np.array([1.0, 0.0, 0.0], dtype=np.float64)
    return np.asarray(vec / norm, dtype=np.float64)


def _quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dtype=np.float64,
    )


def _quat_conj(q: np.ndarray) -> np.ndarray:
    return np.array([q[0], -q[1], -q[2], -q[3]], dtype=np.float64)


def _rotate_body_to_world(quat_wxyz: np.ndarray, vec_body: np.ndarray) -> np.ndarray:
    pure = np.array([0.0, vec_body[0], vec_body[1], vec_body[2]], dtype=np.float64)
    rotated = _quat_mul(_quat_mul(quat_wxyz, pure), _quat_conj(quat_wxyz))
    return rotated[1:]


def _rotation_matrix_to_quat(rot: np.ndarray) -> np.ndarray:
    trace = float(np.trace(rot))
    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        quat = np.array(
            [
                0.25 * s,
                (rot[2, 1] - rot[1, 2]) / s,
                (rot[0, 2] - rot[2, 0]) / s,
                (rot[1, 0] - rot[0, 1]) / s,
            ],
            dtype=np.float64,
        )
    else:
        diag = np.diag(rot)
        idx = int(np.argmax(diag))
        if idx == 0:
            s = np.sqrt(1.0 + rot[0, 0] - rot[1, 1] - rot[2, 2]) * 2.0
            quat = np.array(
                [
                    (rot[2, 1] - rot[1, 2]) / s,
                    0.25 * s,
                    (rot[0, 1] + rot[1, 0]) / s,
                    (rot[0, 2] + rot[2, 0]) / s,
                ],
                dtype=np.float64,
            )
        elif idx == 1:
            s = np.sqrt(1.0 + rot[1, 1] - rot[0, 0] - rot[2, 2]) * 2.0
            quat = np.array(
                [
                    (rot[0, 2] - rot[2, 0]) / s,
                    (rot[0, 1] + rot[1, 0]) / s,
                    0.25 * s,
                    (rot[1, 2] + rot[2, 1]) / s,
                ],
                dtype=np.float64,
            )
        else:
            s = np.sqrt(1.0 + rot[2, 2] - rot[0, 0] - rot[1, 1]) * 2.0
            quat = np.array(
                [
                    (rot[1, 0] - rot[0, 1]) / s,
                    (rot[0, 2] + rot[2, 0]) / s,
                    (rot[1, 2] + rot[2, 1]) / s,
                    0.25 * s,
                ],
                dtype=np.float64,
            )
    return _normalize(quat)


def _rotation_matrix_to_quat_jax(rot: jnp.ndarray) -> jnp.ndarray:
    trace = jnp.trace(rot)

    def positive_branch(r: jnp.ndarray) -> jnp.ndarray:
        s = jnp.sqrt(trace + 1.0) * 2.0
        return jnp.array(
            [
                0.25 * s,
                (r[2, 1] - r[1, 2]) / s,
                (r[0, 2] - r[2, 0]) / s,
                (r[1, 0] - r[0, 1]) / s,
            ]
        )

    def negative_branch(r: jnp.ndarray) -> jnp.ndarray:
        diag = jnp.diag(r)
        idx = jnp.argmax(diag)

        def case0(_: None) -> jnp.ndarray:
            s = jnp.sqrt(1.0 + r[0, 0] - r[1, 1] - r[2, 2]) * 2.0
            return jnp.array(
                [
                    (r[2, 1] - r[1, 2]) / s,
                    0.25 * s,
                    (r[0, 1] + r[1, 0]) / s,
                    (r[0, 2] + r[2, 0]) / s,
                ]
            )

        def case1(_: None) -> jnp.ndarray:
            s = jnp.sqrt(1.0 + r[1, 1] - r[0, 0] - r[2, 2]) * 2.0
            return jnp.array(
                [
                    (r[0, 2] - r[2, 0]) / s,
                    (r[0, 1] + r[1, 0]) / s,
                    0.25 * s,
                    (r[1, 2] + r[2, 1]) / s,
                ]
            )

        def case2(_: None) -> jnp.ndarray:
            s = jnp.sqrt(1.0 + r[2, 2] - r[0, 0] - r[1, 1]) * 2.0
            return jnp.array(
                [
                    (r[1, 0] - r[0, 1]) / s,
                    (r[0, 2] + r[2, 0]) / s,
                    (r[1, 2] + r[2, 1]) / s,
                    0.25 * s,
                ]
            )

        return jax.lax.switch(idx, (case0, case1, case2), operand=None)

    quat = jax.lax.cond(trace > 0.0, positive_branch, negative_branch, rot)
    norm = jnp.linalg.norm(quat)
    safe_norm = jnp.where(norm <= 1e-9, 1.0, norm)
    default_quat = jnp.array([1.0, 0.0, 0.0, 0.0], dtype=quat.dtype)
    return jnp.where((norm <= 1e-9)[None], default_quat, quat / safe_norm)


def _normalize_jax(vec: jnp.ndarray) -> jnp.ndarray:
    norm = jnp.linalg.norm(vec)
    safe_norm = jnp.where(norm <= 1e-9, 1.0, norm)
    default_vec = jnp.array([1.0, 0.0, 0.0], dtype=vec.dtype)
    return jnp.where((norm <= 1e-9)[None], default_vec, vec / safe_norm)


def _quat_mul_jax(q1: jnp.ndarray, q2: jnp.ndarray) -> jnp.ndarray:
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return jnp.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dtype=q1.dtype,
    )


def _quat_conj_jax(q: jnp.ndarray) -> jnp.ndarray:
    return jnp.array([q[0], -q[1], -q[2], -q[3]], dtype=q.dtype)


def _rotate_body_to_world_jax(quat_wxyz: jnp.ndarray, vec_body: jnp.ndarray) -> jnp.ndarray:
    pure = jnp.array([0.0, vec_body[0], vec_body[1], vec_body[2]], dtype=quat_wxyz.dtype)
    rotated = _quat_mul_jax(_quat_mul_jax(quat_wxyz, pure), _quat_conj_jax(quat_wxyz))
    return rotated[1:]


def build_estimator_init_state_jax(dataset, init_idx: int = 0):
    i = jnp.minimum(jnp.maximum(jnp.asarray(init_idx, dtype=jnp.int32), dataset.first_valid_idx), dataset.num_samples - 1)
    alpha_meas = _normalize_jax(dataset.accel[i])
    beta_meas = _normalize_jax(dataset.mag_body[i])
    world_alpha = _normalize_jax(_rotate_body_to_world_jax(dataset.true_quat[i], alpha_meas))
    world_beta = _normalize_jax(_rotate_body_to_world_jax(dataset.true_quat[i], beta_meas))
    config = default_config(world_alpha, world_beta)
    observer_state = init_state(
        alpha_hat0=alpha_meas,
        beta_hat0=beta_meas,
        b_hat0=jnp.zeros(3, dtype=jnp.float32),
        b_alpha_hat0=jnp.zeros(3, dtype=jnp.float32),
        mode0=0,
        config=config,
    )
    state = JaxEstimatorRuntimeState(
        attitude_quat=dataset.true_quat[i],
        bias=jnp.zeros(3, dtype=jnp.float32),
        accel_bias=jnp.zeros(3, dtype=jnp.float32),
        observer_state=observer_state,
    )
    return config, state


@partial(jax.jit, static_argnames=("dt",))
def _observer_step_device(config, observer_state, gyro, accel, mag, mode: int, dt: float):
    next_state, output = observer_step(
        observer_state,
        {
            "omega_m": gyro,
            "alpha_m": accel,
            "beta_m": mag,
        },
        mode_to_one_hot(mode),
        dt,
        config,
    )
    quat = _rotation_matrix_to_quat_jax(output.r_hat)
    return next_state, quat, output.b_hat, output.b_alpha_hat


@partial(jax.jit, static_argnames=("dt",))
def step_estimator_state_jax(
    est_state: JaxEstimatorRuntimeState,
    gyro: jnp.ndarray,
    accel: jnp.ndarray,
    mag: jnp.ndarray,
    mode: jnp.ndarray,
    dt: float,
    config,
) -> JaxEstimatorRuntimeState:
    next_state, quat, bias, accel_bias = _observer_step_device(
        config,
        est_state.observer_state,
        gyro,
        _normalize_jax(accel),
        _normalize_jax(mag),
        mode,
        dt,
    )
    return JaxEstimatorRuntimeState(
        attitude_quat=quat,
        bias=bias,
        accel_bias=accel_bias,
        observer_state=next_state,
    )


@dataclass(frozen=True)
class JaxEstimatorBridge:
    dt: float = 1.0 / 400.0

    def init(self, traj: TrajectoryDataset, init_idx: int = 0) -> EstimatorState:
        if traj.mag_body is None:
            raise ValueError("Trajectory dataset must include mag_body for the switched observer.")
        quat_norm = np.linalg.norm(traj.true_quat, axis=1)
        accel_norm = np.linalg.norm(traj.accel, axis=1)
        valid = np.isfinite(quat_norm) & (quat_norm > 0.5) & np.isfinite(accel_norm) & (accel_norm > 1e-3)
        if not np.any(valid):
            raise ValueError("Dataset does not contain any valid initialization sample.")
        first_valid_idx = int(np.argmax(valid))
        i = min(max(init_idx, first_valid_idx), traj.num_samples - 1)
        alpha_meas = _normalize(traj.accel[i])
        beta_meas = _normalize(traj.mag_body[i])
        world_alpha = _normalize(_rotate_body_to_world(traj.true_quat[i], alpha_meas))
        world_beta = _normalize(_rotate_body_to_world(traj.true_quat[i], beta_meas))
        config = default_config(world_alpha, world_beta)
        observer_state = init_state(
            alpha_hat0=alpha_meas,
            beta_hat0=beta_meas,
            b_hat0=np.zeros(3, dtype=np.float64),
            b_alpha_hat0=np.zeros(3, dtype=np.float64),
            mode0=0,
            config=config,
        )
        return EstimatorState(
            attitude_quat=np.asarray(traj.true_quat[i], dtype=np.float64),
            bias=np.zeros(3, dtype=np.float64),
            accel_bias=np.zeros(3, dtype=np.float64),
            internal_state=(config, observer_state),
        )

    def step(
        self,
        est_state: EstimatorState,
        gyro: np.ndarray,
        accel: np.ndarray,
        mag: np.ndarray | None,
        cmd_rates: np.ndarray,
        throttle: np.ndarray,
        mode: int,
    ) -> EstimatorState:
        del cmd_rates, throttle
        if mag is None:
            raise ValueError("Estimator step requires magnetometer measurements.")
        config, observer_state = est_state.internal_state
        next_state, quat, bias, accel_bias = _observer_step_device(
            config,
            observer_state,
            jnp.asarray(gyro, dtype=jnp.float32),
            jnp.asarray(_normalize(np.asarray(accel, dtype=np.float64)), dtype=jnp.float32),
            jnp.asarray(_normalize(np.asarray(mag, dtype=np.float64)), dtype=jnp.float32),
            int(mode),
            self.dt,
        )
        return EstimatorState(
            attitude_quat=quat,
            bias=bias,
            accel_bias=accel_bias,
            internal_state=(config, next_state),
        )
