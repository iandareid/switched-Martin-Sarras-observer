"""Trajectory segment primitives."""

from __future__ import annotations

import numpy as np


def smoothstep_quintic(u: np.ndarray) -> np.ndarray:
    return 10.0 * u**3 - 15.0 * u**4 + 6.0 * u**5


def segment_hover(
    t_start: float,
    duration: float,
    dt: float,
    pos: np.ndarray,
    yaw: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = int(np.round(duration / dt))
    t_rel = np.arange(n, dtype=np.float64) * dt
    t = t_start + t_rel
    p = np.tile(np.asarray(pos, dtype=np.float64), (n, 1))
    y = np.full((n,), float(yaw), dtype=np.float64)
    return t, p, y


def segment_smooth_move(
    t_start: float,
    duration: float,
    dt: float,
    start_pos: np.ndarray,
    end_pos: np.ndarray,
    start_yaw: float,
    end_yaw: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = int(np.round(duration / dt))
    t_rel = np.arange(n, dtype=np.float64) * dt
    u = np.clip(t_rel / max(duration, 1e-6), 0.0, 1.0)
    s = smoothstep_quintic(u)
    t = t_start + t_rel
    p0 = np.asarray(start_pos, dtype=np.float64)
    p1 = np.asarray(end_pos, dtype=np.float64)
    p = p0[None, :] + (p1 - p0)[None, :] * s[:, None]
    y = start_yaw + (end_yaw - start_yaw) * s
    return t, p, y


def segment_arc_move(
    t_start: float,
    duration: float,
    dt: float,
    center: np.ndarray,
    radius: float,
    yaw_offset: float,
    z_ned: float,
    clockwise: bool,
    start_z_ned: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = int(np.round(duration / dt))
    t_rel = np.arange(n, dtype=np.float64) * dt
    u = np.clip(t_rel / max(duration, 1e-6), 0.0, 1.0)
    dtheta = (np.pi / 2.0) * (1.0 if clockwise else -1.0)
    s = smoothstep_quintic(u)
    theta = yaw_offset + dtheta * s
    t = t_start + t_rel
    c = np.asarray(center, dtype=np.float64)
    c0 = np.array([np.cos(yaw_offset), np.sin(yaw_offset)], dtype=np.float64)
    z0 = float(z_ned if start_z_ned is None else start_z_ned)
    z = z0 + (float(z_ned) - z0) * s
    p = np.column_stack(
        [
            c[0] + radius * (np.cos(theta) - c0[0]),
            c[1] + radius * (np.sin(theta) - c0[1]),
            z,
        ]
    )
    # Align yaw roughly with tangent.
    tangent = theta + (np.pi / 2.0) * (1.0 if clockwise else -1.0)
    y = tangent
    return t, p, y
