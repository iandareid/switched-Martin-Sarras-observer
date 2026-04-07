"""Batch trajectory generation utilities."""

from __future__ import annotations

import numpy as np

from trajgen.profiles import HumanLikeProfile, generate_human_like
from trajgen.types import BatchedTrajectory, unwrap_yaw, wrap_angle


def generate_batch(
    seeds: np.ndarray,
    dt: float,
    duration: float,
    profile: HumanLikeProfile | None = None,
) -> BatchedTrajectory:
    seed_arr = np.asarray(seeds, dtype=np.int64).reshape(-1)
    if seed_arr.size == 0:
        raise ValueError("seeds must be non-empty.")

    trajs = [generate_human_like(int(seed), dt=dt, duration=duration, profile=profile) for seed in seed_arr]
    t = np.arange(0.0, duration + 1e-9, dt, dtype=np.float64)

    pos_list = []
    vel_list = []
    acc_list = []
    yaw_list = []
    yaw_rate_list = []
    for tr in trajs:
        pos_i, vel_i, acc_i, yaw_i = tr.sample_batch(t)
        yaw_unwrapped = unwrap_yaw(yaw_i)
        yaw_rate_i = np.gradient(yaw_unwrapped, t, edge_order=2)
        pos_list.append(pos_i)
        vel_list.append(vel_i)
        acc_list.append(acc_i)
        yaw_list.append(wrap_angle(yaw_i))
        yaw_rate_list.append(yaw_rate_i)

    pos = np.stack(pos_list, axis=0)
    yaw = np.stack(yaw_list, axis=0)
    vel = np.stack(vel_list, axis=0)
    acc = np.stack(acc_list, axis=0)
    yaw_rate = np.stack(yaw_rate_list, axis=0)
    return BatchedTrajectory(t=t, pos=pos, yaw=yaw, vel=vel, acc=acc, yaw_rate=yaw_rate)
