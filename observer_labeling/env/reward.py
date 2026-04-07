from __future__ import annotations

import jax.numpy as jnp

MODE_GO = 0
MODE_BVO = 1
MODE_PAE = 2
WORLD_GRAVITY_DIR = jnp.array([0.0, 0.0, -1.0], dtype=jnp.float32)


def quat_angle_error(true_quat, est_quat) -> jnp.ndarray:
    dot = jnp.abs(jnp.dot(jnp.asarray(true_quat), jnp.asarray(est_quat)))
    dot = jnp.clip(dot, -1.0, 1.0)
    return 2.0 * jnp.arccos(dot)


def bias_l2_error(true_bias, est_bias) -> jnp.ndarray:
    if true_bias is None or est_bias is None:
        return jnp.asarray(0.0, dtype=jnp.float32)
    return jnp.linalg.norm(jnp.asarray(true_bias) - jnp.asarray(est_bias))


def smoothstep01(value) -> jnp.ndarray:
    x = jnp.clip(jnp.asarray(value, dtype=jnp.float32), 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def scaled_error_norm(
    error_norm,
    pre_break_gain: float,
    break_norm: float,
    transition_width: float,
    post_break_gain: float,
) -> jnp.ndarray:
    error = jnp.asarray(error_norm, dtype=jnp.float32)
    low_gain = jnp.asarray(pre_break_gain, dtype=jnp.float32)
    high_gain = jnp.asarray(post_break_gain, dtype=jnp.float32)
    knee = jnp.asarray(break_norm, dtype=jnp.float32)
    width = jnp.asarray(transition_width, dtype=jnp.float32)
    use_step = width <= 0.0
    t_smooth = (error - knee) / jnp.maximum(width, 1e-6)
    t = jnp.where(use_step, jnp.where(error >= knee, 1.0, 0.0), t_smooth)
    gain = low_gain + smoothstep01(t) * (high_gain - low_gain)
    return gain * error


def scaled_accel_bias_error(
    true_bias,
    est_bias,
    pre_break_gain: float,
    break_norm: float,
    transition_width: float,
    post_break_gain: float,
) -> jnp.ndarray:
    error_norm = bias_l2_error(true_bias, est_bias)
    return scaled_error_norm(
        error_norm,
        pre_break_gain,
        break_norm,
        transition_width,
        post_break_gain,
    )


def exponential_decay_sequence(initial_error, convergence_steps: int, target_fraction: float = 0.01) -> jnp.ndarray:
    error0 = jnp.asarray(initial_error, dtype=jnp.float32)
    steps = int(convergence_steps)
    if steps <= 0:
        return jnp.zeros((0,), dtype=jnp.float32)
    safe_fraction = jnp.clip(jnp.asarray(target_fraction, dtype=jnp.float32), 1e-6, 1.0)
    rate = -jnp.log(safe_fraction) / jnp.maximum(jnp.asarray(float(steps), dtype=jnp.float32), 1.0)
    indices = jnp.arange(steps, dtype=jnp.float32)
    return error0 * jnp.exp(-rate * indices)


def normalize_vec(vec, eps: float = 1e-6) -> jnp.ndarray:
    arr = jnp.asarray(vec, dtype=jnp.float32)
    norm = jnp.linalg.norm(arr)
    safe_norm = jnp.maximum(norm, jnp.asarray(eps, dtype=arr.dtype))
    return arr / safe_norm


def cross_matrix(vec) -> jnp.ndarray:
    x, y, z = jnp.asarray(vec, dtype=jnp.float32)
    return jnp.array(
        [
            [0.0, -z, y],
            [z, 0.0, -x],
            [-y, x, 0.0],
        ],
        dtype=jnp.float32,
    )


def beta_pe_matrix(beta) -> jnp.ndarray:
    skew = cross_matrix(normalize_vec(beta))
    return -(skew @ skew)


def clipped_signed_margin(margin, clip_value: float) -> jnp.ndarray:
    margin_arr = jnp.asarray(margin, dtype=jnp.float32)
    clip_arr = jnp.asarray(clip_value, dtype=jnp.float32)
    return jnp.clip(margin_arr, -clip_arr, clip_arr)


def mag_pe_margin(mean_pe_matrix, mu: float) -> jnp.ndarray:
    mean_pe_matrix = jnp.asarray(mean_pe_matrix, dtype=jnp.float32)
    eigvals = jnp.linalg.eigvalsh(mean_pe_matrix)
    return eigvals[0] - jnp.asarray(mu, dtype=jnp.float32)


def quat_conj(quat) -> jnp.ndarray:
    q = jnp.asarray(quat, dtype=jnp.float32)
    return jnp.array([q[0], -q[1], -q[2], -q[3]], dtype=q.dtype)


def quat_mul(q1, q2) -> jnp.ndarray:
    a = jnp.asarray(q1, dtype=jnp.float32)
    b = jnp.asarray(q2, dtype=jnp.float32)
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return jnp.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dtype=a.dtype,
    )


def rotate_world_to_body(quat_wxyz, vec_world) -> jnp.ndarray:
    quat = normalize_vec(quat_wxyz)
    pure = jnp.asarray([0.0, vec_world[0], vec_world[1], vec_world[2]], dtype=quat.dtype)
    rotated = quat_mul(quat_mul(quat_conj(quat), pure), quat)
    return rotated[1:]


def gravity_direction_body(true_quat) -> jnp.ndarray:
    return normalize_vec(rotate_world_to_body(true_quat, WORLD_GRAVITY_DIR))


def accel_gravity_alignment(accel, true_quat) -> jnp.ndarray:
    accel_dir = normalize_vec(accel)
    # The recorded accelerometer is specific force, so in near-hover conditions
    # it points opposite the gravity direction expressed in body coordinates.
    gravity_ref_dir = -gravity_direction_body(true_quat)
    return jnp.clip(jnp.dot(accel_dir, gravity_ref_dir), -1.0, 1.0)


def accel_gravity_margin(mean_alignment, close_angle_rad: float) -> jnp.ndarray:
    alignment = jnp.clip(jnp.asarray(mean_alignment, dtype=jnp.float32), -1.0, 1.0)
    angle = jnp.arccos(alignment)
    return jnp.asarray(close_angle_rad, dtype=jnp.float32) - angle
