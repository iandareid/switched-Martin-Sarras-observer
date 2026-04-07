from __future__ import annotations

import math

from .backend import jnp

kPi = math.pi


def _expand_last(mask, ndim):
    out = mask
    while out.ndim < ndim:
        out = out[..., None]
    return out


def select_by_mask(mask, on_true, on_false):
    mask = _expand_last(mask, on_true.ndim)
    return jnp.where(mask, on_true, on_false)


def safe_normalize(v, eps):
    norm = jnp.linalg.norm(v, axis=-1, keepdims=True)
    denom = jnp.where(norm > eps, norm, 1.0)
    normalized = v / denom
    valid = norm > eps
    return normalized, valid


def skew(v):
    zero = jnp.zeros_like(v[..., 0])
    return jnp.stack(
        [
            jnp.stack([zero, -v[..., 2], v[..., 1]], axis=-1),
            jnp.stack([v[..., 2], zero, -v[..., 0]], axis=-1),
            jnp.stack([-v[..., 1], v[..., 0], zero], axis=-1),
        ],
        axis=-2,
    )


def make_reference_basis(alpha_i, beta_i, eps_collinear=1e-9, eps_norm=1e-12):
    e1, valid_alpha = safe_normalize(alpha_i, eps_norm)
    alpha_cross_beta = cross(alpha_i, beta_i)
    cross_norm = jnp.linalg.norm(alpha_cross_beta, axis=-1, keepdims=True)
    e2 = alpha_cross_beta / jnp.where(cross_norm > eps_collinear, cross_norm, 1.0)
    third = cross(alpha_i, alpha_cross_beta)
    e3, valid_e3 = safe_normalize(third, eps_norm)
    basis = jnp.stack([e1, e2, e3], axis=-1)
    valid = jnp.squeeze(valid_alpha & (cross_norm > eps_collinear) & valid_e3, axis=-1)
    eye = jnp.broadcast_to(jnp.eye(3), basis.shape)
    return jnp.where(valid[..., None, None], basis, eye)


def cross(a, b):
    return jnp.cross(a, b, axis=-1)


def reconstruct_rotation(alpha_hat, beta_hat, reference_basis, eps_collinear=1e-9, eps_norm=1e-12):
    alpha_n, valid_alpha = safe_normalize(alpha_hat, eps_norm)
    alpha_cross_beta = cross(alpha_hat, beta_hat)
    cross_norm = jnp.linalg.norm(alpha_cross_beta, axis=-1, keepdims=True)
    second = alpha_cross_beta / jnp.where(cross_norm > eps_collinear, cross_norm, 1.0)
    third_raw = cross(alpha_hat, alpha_cross_beta)
    third, valid_third = safe_normalize(third_raw, eps_norm)
    estimate_basis = jnp.stack([alpha_n, second, third], axis=-1)
    r_candidate = jnp.matmul(reference_basis, jnp.swapaxes(estimate_basis, -1, -2))
    valid = jnp.squeeze(valid_alpha & (cross_norm > eps_collinear) & valid_third, axis=-1)
    eye = jnp.broadcast_to(jnp.eye(3), r_candidate.shape)
    r_hat = jnp.where(valid[..., None, None], r_candidate, eye)
    return r_hat, valid


def matrix_to_euler321(r):
    pitch = jnp.arcsin(jnp.clip(-r[..., 2, 0], -1.0, 1.0))
    roll = jnp.arctan2(r[..., 2, 1], r[..., 2, 2])
    yaw = jnp.arctan2(r[..., 1, 0], r[..., 0, 0])
    return jnp.stack([roll, pitch, yaw], axis=-1)


def rotation_error_deg(estimate, truth):
    error = jnp.matmul(estimate, jnp.swapaxes(truth, -1, -2))
    arg = jnp.clip((jnp.trace(error, axis1=-2, axis2=-1) - 1.0) * 0.5, -1.0, 1.0)
    return jnp.arccos(arg) * (180.0 / kPi)
