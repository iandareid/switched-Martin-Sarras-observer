"""Trajectory profile generators."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from trajgen.composer import new_series
from trajgen.segments import segment_arc_move, segment_hover, segment_smooth_move
from trajgen.types import TrajectoryBuffer, wrap_angle


@dataclass
class HumanLikeProfile:
    hover_duration_range: tuple[float, float] = (0.4, 3.0)
    move_duration_range: tuple[float, float] = (1.5, 4.0)
    speed_limit: float = 8.0
    accel_limit: float = 3.5
    yaw_rate_limit: float = np.deg2rad(80.0)
    z_range: tuple[float, float] = (-10.0, -0.7)  # NED, negative is above ground.
    complex_ratio: float = 0.5
    move_count_range: tuple[int, int] = (5, 6)
    final_hover_duration: float = 3.0
    area_n: tuple[float, float] = (-60.0, 60.0)
    area_e: tuple[float, float] = (-60.0, 60.0)


@dataclass
class HumanLikeSegment:
    kind: str
    t_start: float
    t_end: float
    start_pos: np.ndarray
    end_pos: np.ndarray
    start_yaw: float
    end_yaw: float


def _apply_yaw_rate_limit(
    yaw: np.ndarray, dt: float, yaw_rate_limit: float, yaw0: float | None = None
) -> np.ndarray:
    if yaw.size <= 1:
        return yaw
    y_raw = np.asarray(yaw, dtype=np.float64)
    if yaw0 is None:
        y_target = np.unwrap(y_raw)
        y_start = y_target[0]
    else:
        y_concat = np.unwrap(np.concatenate([np.array([yaw0], dtype=np.float64), y_raw]))
        y_start = y_concat[0]
        y_target = y_concat[1:]
    max_step = max(float(yaw_rate_limit) * dt, 0.0)
    y_limited = np.empty_like(y_target)
    y_limited[0] = y_start
    for i in range(1, y_target.shape[0]):
        dy = y_target[i] - y_limited[i - 1]
        y_limited[i] = y_limited[i - 1] + np.clip(dy, -max_step, max_step)
    return wrap_angle(y_limited)


def generate_default_demo(dt: float = 0.05) -> TrajectoryBuffer:
    """2s hover at (0,0,1m up) -> 4s smooth move north 5m -> 2s hover."""
    s = new_series()
    start = np.array([0.0, 0.0, -1.0], dtype=np.float64)
    end = np.array([5.0, 0.0, -1.0], dtype=np.float64)

    t0, p0, y0 = segment_hover(0.0, 2.0, dt, start, 0.0)
    t1, p1, y1 = segment_smooth_move(t0[-1] + dt, 4.0, dt, start, end, 0.0, 0.0)
    t2, p2, y2 = segment_hover(t1[-1] + dt, 2.0, dt, end, 0.0)

    s.append(t0, p0, y0)
    s.append(t1, p1, y1)
    s.append(t2, p2, y2)
    return s.build(dt=dt)


def _bounded_target(rng: np.random.Generator, p: HumanLikeProfile, current: np.ndarray) -> np.ndarray:
    cand = np.array(
        [
            rng.uniform(*p.area_n),
            rng.uniform(*p.area_e),
            rng.uniform(*p.z_range),
        ],
        dtype=np.float64,
    )
    delta = cand - current
    horiz = np.linalg.norm(delta[:2])
    if horiz < 1e-6:
        return cand
    return cand


def _required_duration_for_path(path_length: float, p: HumanLikeProfile) -> float:
    # For quintic smoothstep s(u):
    # max |s'(u)| = 1.875 and max |s''(u)| = sqrt(100/3) ~= 5.7735
    vmax = max(float(p.speed_limit), 1e-6)
    amax = max(float(p.accel_limit), 1e-6)
    t_speed = 1.875 * path_length / vmax
    t_accel = np.sqrt(5.773502691896258 * path_length / amax)
    return max(t_speed, t_accel, 0.0)


def _max_path_for_duration(duration: float, p: HumanLikeProfile) -> float:
    if duration <= 0.0:
        return 0.0
    vmax = max(float(p.speed_limit), 0.0)
    amax = max(float(p.accel_limit), 0.0)
    by_speed = vmax * duration / 1.875 if vmax > 0.0 else 0.0
    by_accel = amax * duration * duration / 5.773502691896258 if amax > 0.0 else 0.0
    return min(by_speed, by_accel)


def generate_human_like(
    seed: int,
    dt: float,
    duration: float,
    profile: HumanLikeProfile | None = None,
) -> TrajectoryBuffer:
    traj, _ = generate_human_like_with_plan(seed=seed, dt=dt, duration=duration, profile=profile)
    return traj


def generate_human_like_with_plan(
    seed: int,
    dt: float,
    duration: float,
    profile: HumanLikeProfile | None = None,
) -> tuple[TrajectoryBuffer, list[HumanLikeSegment]]:
    p = profile or HumanLikeProfile()
    rng = np.random.default_rng(seed)
    s = new_series()
    segments: list[HumanLikeSegment] = []

    t_cursor = 0.0
    pos = np.array([0.0, 0.0, -1.0], dtype=np.float64)
    yaw = 0.0
    target_moves = int(rng.integers(p.move_count_range[0], p.move_count_range[1] + 1))
    final_hover_dur = min(max(p.final_hover_duration, 0.0), duration)
    final_hover_start = max(0.0, duration - final_hover_dur)
    min_move_dur = max(min(p.move_duration_range), dt)

    for move_idx in range(target_moves):
        if t_cursor >= final_hover_start - dt:
            break

        moves_left = target_moves - move_idx
        budget_before_final_hover = max(final_hover_start - t_cursor, 0.0)
        max_hover_dur = budget_before_final_hover - moves_left * min_move_dur
        if max_hover_dur > dt:
            hover_dur = float(rng.uniform(*p.hover_duration_range))
            hover_dur = min(hover_dur, max_hover_dur)
            hover_start_pos = pos.copy()
            hover_start_yaw = yaw
            th, ph, yh = segment_hover(t_cursor, hover_dur, dt, pos, yaw)
            s.append(th, ph, yh)
            if th.size > 0:
                segments.append(
                    HumanLikeSegment(
                        kind="hover",
                        t_start=float(th[0]),
                        t_end=float(th[-1] + dt),
                        start_pos=hover_start_pos,
                        end_pos=ph[-1].copy(),
                        start_yaw=float(hover_start_yaw),
                        end_yaw=float(yh[-1]),
                    )
                )
                t_cursor = th[-1] + dt

        moves_after_this = target_moves - move_idx - 1
        max_move_dur = final_hover_start - t_cursor - moves_after_this * min_move_dur
        if max_move_dur <= dt:
            break

        move_dur = float(rng.uniform(*p.move_duration_range))
        move_dur = min(move_dur, max_move_dur)
        use_complex = bool(rng.random() < p.complex_ratio)
        target = _bounded_target(rng, p, pos)

        if use_complex:
            radius = float(rng.uniform(1.0, 4.0))
            clockwise = bool(rng.integers(0, 2))
            theta_total = np.pi / 2.0
            max_path = _max_path_for_duration(max_move_dur, p)
            max_radius = max_path / max(theta_total, 1e-6)
            radius = min(radius, max(max_radius, 0.2))
            path_len = radius * theta_total
            move_dur = min(max(move_dur, _required_duration_for_path(path_len, p)), max_move_dur)
            arc_center = np.array([pos[0], pos[1]], dtype=np.float64)
            arc_start_pos = pos.copy()
            arc_start_yaw = yaw
            ta, pa, ya = segment_arc_move(
                t_start=t_cursor,
                duration=move_dur,
                dt=dt,
                center=arc_center,
                radius=radius,
                yaw_offset=yaw,
                z_ned=float(target[2]),
                start_z_ned=float(pos[2]),
                clockwise=clockwise,
            )
            ya = _apply_yaw_rate_limit(ya, dt=dt, yaw_rate_limit=p.yaw_rate_limit, yaw0=yaw)
            s.append(ta, pa, ya)
            if ta.size > 0:
                arc_kind = "arc_cw" if clockwise else "arc_ccw"
                segments.append(
                    HumanLikeSegment(
                        kind=arc_kind,
                        t_start=float(ta[0]),
                        t_end=float(ta[-1] + dt),
                        start_pos=arc_start_pos,
                        end_pos=pa[-1].copy(),
                        start_yaw=float(arc_start_yaw),
                        end_yaw=float(ya[-1]),
                    )
                )
                pos = pa[-1].copy()
                yaw = wrap_angle(np.array([ya[-1]], dtype=np.float64))[0]
                t_cursor = ta[-1] + dt
        else:
            # Constrain duration and displacement to kinematic limits.
            delta = target - pos
            dist = np.linalg.norm(delta)
            max_path = _max_path_for_duration(max_move_dur, p)
            if dist > max_path and dist > 1e-9:
                delta = delta * (max_path / dist)
                target = pos + delta
                dist = max_path
            min_dur = _required_duration_for_path(dist, p)
            eff_dur = min(max(move_dur, min_dur), max_move_dur)
            move_start_pos = pos.copy()
            move_start_yaw = yaw
            tm, pm, ym = segment_smooth_move(
                t_start=t_cursor,
                duration=eff_dur,
                dt=dt,
                start_pos=pos,
                end_pos=target,
                start_yaw=yaw,
                end_yaw=float(np.arctan2(delta[1], delta[0])) if np.linalg.norm(delta[:2]) > 1e-6 else yaw,
            )
            ym = _apply_yaw_rate_limit(ym, dt=dt, yaw_rate_limit=p.yaw_rate_limit, yaw0=yaw)
            s.append(tm, pm, ym)
            if tm.size > 0:
                segments.append(
                    HumanLikeSegment(
                        kind="smooth_move",
                        t_start=float(tm[0]),
                        t_end=float(tm[-1] + dt),
                        start_pos=move_start_pos,
                        end_pos=pm[-1].copy(),
                        start_yaw=float(move_start_yaw),
                        end_yaw=float(ym[-1]),
                    )
                )
                pos = pm[-1].copy()
                yaw = wrap_angle(np.array([ym[-1]], dtype=np.float64))[0]
                t_cursor = tm[-1] + dt

    final_hold = duration - t_cursor
    if final_hold > dt:
        tf, pf, yf = segment_hover(t_cursor, final_hold, dt, pos, yaw)
        s.append(tf, pf, yf)
        if tf.size > 0:
            segments.append(
                HumanLikeSegment(
                    kind="final_hover",
                    t_start=float(tf[0]),
                    t_end=float(tf[-1] + dt),
                    start_pos=pos.copy(),
                    end_pos=pf[-1].copy(),
                    start_yaw=float(yaw),
                    end_yaw=float(yf[-1]),
                )
            )

    return s.build(dt=dt), segments
