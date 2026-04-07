from __future__ import annotations

from functools import partial

from flax import struct
import jax
import jax.numpy as jnp


@struct.dataclass
class MahonyParams:
    gravity_ref_world: jnp.ndarray
    mag_ref_world: jnp.ndarray
    k_p: float = 2.0
    k_i: float = 0.1
    accel_weight: float = 1.0
    mag_weight: float = 1.0
    accel_gate_margin_mps2: float = 1.0
    g_ref_mps2: float = 9.81
    eps: float = 1e-6


@struct.dataclass
class MahonyState:
    attitude_quat: jnp.ndarray
    gyro_bias: jnp.ndarray


def _normalize_quat_jax(quat: jnp.ndarray, eps: float) -> jnp.ndarray:
    norm = jnp.linalg.norm(quat)
    safe_norm = jnp.where(norm <= eps, 1.0, norm)
    default_quat = jnp.array([1.0, 0.0, 0.0, 0.0], dtype=quat.dtype)
    return jnp.where((norm <= eps)[None], default_quat, quat / safe_norm)


def _normalize_vec_jax(vec: jnp.ndarray, eps: float, default_vec: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    norm = jnp.linalg.norm(vec)
    safe_norm = jnp.where(norm <= eps, 1.0, norm)
    normalized = vec / safe_norm
    return jnp.where((norm <= eps)[None], default_vec, normalized), norm


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


def _rotate_world_to_body_jax(quat_wxyz: jnp.ndarray, vec_world: jnp.ndarray) -> jnp.ndarray:
    return _rotate_body_to_world_jax(_quat_conj_jax(quat_wxyz), vec_world)


def build_mahony_init_state_jax(
    dataset,
    init_idx: int = 0,
    *,
    k_p: float = 2.0,
    k_i: float = 0.1,
    accel_weight: float = 1.0,
    mag_weight: float = 1.0,
    accel_gate_margin_mps2: float = 1.0,
    g_ref_mps2: float = 9.81,
):
    i = jnp.minimum(jnp.maximum(jnp.asarray(init_idx, dtype=jnp.int32), dataset.first_valid_idx), dataset.num_samples - 1)
    default_gravity = jnp.array([0.0, 0.0, -1.0], dtype=jnp.float32)
    default_mag = jnp.array([1.0, 0.0, 0.0], dtype=jnp.float32)
    eps = 1e-6
    accel_body, _ = _normalize_vec_jax(dataset.accel[i], eps, default_gravity)
    mag_body, _ = _normalize_vec_jax(dataset.mag_body[i], eps, default_mag)
    quat0 = _normalize_quat_jax(dataset.true_quat[i], eps)
    params = MahonyParams(
        gravity_ref_world=_normalize_vec_jax(_rotate_body_to_world_jax(quat0, accel_body), eps, default_gravity)[0],
        mag_ref_world=_normalize_vec_jax(_rotate_body_to_world_jax(quat0, mag_body), eps, default_mag)[0],
        k_p=float(k_p),
        k_i=float(k_i),
        accel_weight=float(accel_weight),
        mag_weight=float(mag_weight),
        accel_gate_margin_mps2=float(accel_gate_margin_mps2),
        g_ref_mps2=float(g_ref_mps2),
        eps=eps,
    )
    state = MahonyState(
        attitude_quat=quat0,
        gyro_bias=jnp.zeros(3, dtype=jnp.float32),
    )
    return params, state


@jax.jit
def mahony_accel_correction_enabled(accel: jnp.ndarray, params: MahonyParams) -> jnp.ndarray:
    accel_norm = jnp.linalg.norm(accel)
    return accel_norm <= (jnp.asarray(params.g_ref_mps2, dtype=accel.dtype) + jnp.asarray(params.accel_gate_margin_mps2, dtype=accel.dtype))


@partial(jax.jit, static_argnames=("dt",))
def step_mahony_state_jax(
    state: MahonyState,
    gyro: jnp.ndarray,
    accel: jnp.ndarray,
    mag: jnp.ndarray,
    dt: float,
    params: MahonyParams,
) -> MahonyState:
    default_gravity = jnp.array([0.0, 0.0, -1.0], dtype=gyro.dtype)
    default_mag = jnp.array([1.0, 0.0, 0.0], dtype=gyro.dtype)
    q_hat = _normalize_quat_jax(state.attitude_quat, params.eps)
    accel_meas, _ = _normalize_vec_jax(accel, params.eps, default_gravity)
    mag_meas, _ = _normalize_vec_jax(mag, params.eps, default_mag)
    accel_hat = _normalize_vec_jax(_rotate_world_to_body_jax(q_hat, params.gravity_ref_world), params.eps, default_gravity)[0]
    mag_hat = _normalize_vec_jax(_rotate_world_to_body_jax(q_hat, params.mag_ref_world), params.eps, default_mag)[0]
    accel_enabled = mahony_accel_correction_enabled(accel, params)
    accel_term = jnp.where(
        accel_enabled,
        jnp.asarray(params.accel_weight, dtype=gyro.dtype) * jnp.cross(accel_meas, accel_hat),
        jnp.zeros(3, dtype=gyro.dtype),
    )
    mag_term = jnp.asarray(params.mag_weight, dtype=gyro.dtype) * jnp.cross(mag_meas, mag_hat)
    omega_mes = accel_term + mag_term
    omega_c = gyro - state.gyro_bias + jnp.asarray(params.k_p, dtype=gyro.dtype) * omega_mes
    q_dot = 0.5 * _quat_mul_jax(q_hat, jnp.array([0.0, omega_c[0], omega_c[1], omega_c[2]], dtype=gyro.dtype))
    q_next = _normalize_quat_jax(q_hat + jnp.asarray(dt, dtype=gyro.dtype) * q_dot, params.eps)
    gyro_bias_next = state.gyro_bias - jnp.asarray(dt, dtype=gyro.dtype) * jnp.asarray(params.k_i, dtype=gyro.dtype) * omega_mes
    return MahonyState(attitude_quat=q_next, gyro_bias=gyro_bias_next)
